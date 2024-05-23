import functools
import logging
import random
import string
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Dict as GymDict
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import convert_to_torch_tensor, convert_to_non_torch_type, huber_loss
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

from niql.models import DRQNModel, SimpleCommNet, HyperEncoder
from niql.utils import preprocess_trajectory_batch, unpack_observation, NEIGHBOUR_NEXT_OBS, NEIGHBOUR_OBS, \
    to_numpy, unroll_mac, unroll_mac_squeeze_wrapper, save_representations, mac, get_lds_weights

logger = logging.getLogger(__name__)


def get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


def soft_update(target_net, source_net, tau):
    """
    Soft update the parameters of the target network with those of the source network.

    Args:
    - target_net: Target network.
    - source_net: Source network.
    - tau: Soft update parameter (0 < tau <= 1).

    Returns:
    - target_net: Updated target network.
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    return target_net


def pairwise_l2_distance(tensor):
    # Calculate pairwise L2 distances
    diff = tensor.unsqueeze(1) - tensor.unsqueeze(0)  # Broadcasting to compute differences
    l2_distances = torch.sqrt(torch.sum(diff ** 2, dim=2))  # L2 distance calculation
    return l2_distances


def cosine_embedding_loss(embeddings, labels, margin=0.5):
    """
    Function for the cosine embedding loss.

    Args:
    - embeddings (Tensor): Embedding tensor of shape (N, embedding_dim)
    - labels (Tensor): Tensor of shape (N, 1)

    Returns:
    - loss (Tensor): The computed cosine embedding loss.
    """
    emb_chunks = torch.split(embeddings, 64)
    lbl_chunks = torch.split(labels, 64)

    positive_losses = []
    negative_losses = []

    def sim_loss(emb, lbl):
        # compute abs differences in labels
        labels_xx, labels_yy = torch.meshgrid(lbl, lbl)
        abs_diff = torch.abs(labels_xx - labels_yy)

        # Compute cosine similarities between all pairs
        similarities = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=2)

        # Masks for similar and dissimilar pairs
        positive_mask = abs_diff <= 0.0005
        negative_mask = ~positive_mask

        # Compute loss components
        positive_loss = 1 - similarities[positive_mask]
        negative_loss = torch.clamp(similarities[negative_mask] - margin, min=0.0)
        positive_losses.append(positive_loss)
        negative_losses.append(negative_loss)

    for sub_emb, sub_lbl in zip(emb_chunks, lbl_chunks):
        sim_loss(sub_emb, sub_lbl)

    # Combine loss components
    loss = torch.mean(torch.cat(positive_losses + negative_losses, dim=0))

    return loss


def l2_embedding_loss(embeddings, bucket_indices, margin=2):
    """
    Function for the L2 embedding loss.

    Args:
    - embeddings (Tensor): Embedding tensor of shape (N, embedding_dim)
    - bucket_indices (Tensor): Tensor of shape (N, 1) representing the bucket index of each representation's Q value

    Returns:
    - loss (Tensor): The computed cosine embedding loss.
    """
    B, dim = embeddings.shape

    # construct NxN labels tensor
    labels = torch.zeros((B, B), device=embeddings.device, dtype=torch.float) - 1.
    for i, b_idx in enumerate(bucket_indices):
        labels[i, bucket_indices == b_idx] = 1.

    # Compute cosine similarities between all pairs
    distances = pairwise_l2_distance(embeddings)
    # distances = F.normalize(distances)

    # Masks for similar and dissimilar pairs
    positive_mask = labels == 1.
    negative_mask = labels == -1.

    # Compute loss components
    positive_loss = distances[positive_mask]
    negative_loss = torch.clamp(margin - distances[negative_mask], min=0)

    # Combine loss components
    loss = torch.mean(torch.cat((positive_loss, negative_loss), dim=0))

    return loss


class WBQLPolicy(Policy):
    """
    Implementation of Weighted Best Possible Q-learning
    """

    def __init__(self, obs_space, action_space, config):
        self.framework = "torch"
        config = dict(DEFAULT_CONFIG, **config)
        super().__init__(obs_space, action_space, config)
        self.n_agents = 1
        self.agent_id = "agent_" + "".join(random.choices(string.ascii_letters, k=4))
        config["model"]["n_agents"] = self.n_agents
        self.lamda = config["lambda"]
        self.tau = config["tau"]
        self.n_actions = action_space.n
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        config["model"]["full_obs_space"] = obs_space
        config["model"]["device"] = self.device
        core_arch = config["model"]["custom_model_config"]["model_arch_args"]["core_arch"]
        self.reward_standardize = config["reward_standardize"]
        self.neighbour_messages = []
        self._fds_epoch = 0

        agent_obs_space = obs_space.original_space
        if isinstance(agent_obs_space, GymDict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError("Dict obs space must have subspace labeled `obs`")
            self.obs_size = get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError(
                        "Action mask shape must be {}, got {}".format(
                            (self.n_actions,), mask_shape))
                self.has_action_mask = True
            if "state" in space_keys:
                self.env_global_state_shape = get_size(agent_obs_space.spaces["state"])
                self.has_env_global_state = True
            else:
                self.env_global_state_shape = (self.obs_size, self.n_agents)
            # The real agent obs space is nested inside the dict
            config["model"]["full_obs_space"] = agent_obs_space
            agent_obs_space = agent_obs_space.spaces["obs"]
        else:
            self.obs_size = get_size(agent_obs_space)
            self.env_global_state_shape = (self.obs_size, self.n_agents)

        # models
        self.use_comm = self.config.get("comm_dim", 0) > 0
        if self.use_comm:
            com_dim = self.config["comm_dim"]
            com_hdim = self.config["comm_dim"]
            config["model"]["comm_dim"] = com_dim
            self.comm_net = SimpleCommNet(self.obs_size, com_hdim, com_dim).to(self.device)
            self.comm_net_target = SimpleCommNet(self.obs_size, com_hdim, com_dim).to(self.device)
            self.comm_agg = HyperEncoder(config["model"]).to(self.device)
            self.comm_agg_target = HyperEncoder(config["model"]).to(self.device)

        fds_model_config = dict(config["model"])
        self.fds_config = {
            "bucket_num": 100,
            "kernel": "gaussian",
            "ks": 5,
            "sigma": 2,
            "momentum": 0.9,
            "device": self.device,
        }
        fds_model_config["fds"] = self.fds_config
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            fds_model_config,
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.auxiliary_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.auxiliary_model_target = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.exploration = self._create_exploration()
        self.dist_class = TorchCategorical

        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0
        self._training_iteration_num = 0
        self._global_update_count = 0

        # optimizer
        self.params = list(self.model.parameters()) + list(self.auxiliary_model.parameters())
        if self.use_comm:
            self.params += list(self.comm_net.parameters())
            self.params += list(self.comm_agg.parameters())

        if config["optimizer"] == "rmsprop":
            from torch.optim import RMSprop
            self.optimiser = RMSprop(params=self.params, lr=config["lr"])

        elif config["optimizer"] == "adam":
            from torch.optim import Adam
            self.optimiser = Adam(params=self.params, lr=config["lr"])

        else:
            raise ValueError("choose one optimizer type from rmsprop(RMSprop) or adam(Adam)")

        # Auto-update model's inference view requirements, if recurrent.
        self._update_model_view_requirements_from_init_state()
        # Combine view_requirements for Model and Policy.
        self.view_requirements.update(self.model.view_requirements)

        # initial target network sync
        self.update_target()

    @override(Policy)
    def compute_actions(
            self,
            obs_batch: Union[List[TensorStructType], TensorStructType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
            prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep
        obs_batch, action_mask, _ = unpack_observation(self, obs_batch)

        # Switch to eval mode.
        if self.model:
            self.model.eval()
            if self.use_comm:
                self.comm_net.eval()
                self.comm_agg.eval()

        with torch.no_grad():
            obs_batch = convert_to_torch_tensor(obs_batch, self.device)
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]

            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)

            if self.use_comm:
                msg = self.comm_net(obs_batch)
                if self.neighbour_messages:
                    n = len(self.neighbour_messages)
                    local_msg = msg.reshape(1, 1, -1)
                    neighbour_msgs = convert_to_torch_tensor(
                        np.array(self.neighbour_messages).reshape(1, n, -1), self.device)
                    msg = torch.cat([local_msg, neighbour_msgs], dim=1)
                    msg = self.aggregate_messages(msg)
                    self.neighbour_messages.clear()
                obs_batch = torch.cat([obs_batch, msg], dim=-1)

            # if self.config["use_obs_encoder"]:
            #     obs_batch = convert_to_torch_tensor(obs_batch, self.device)
            #     obs_batch, _ = self.obs_encoder(obs_batch)
            # else:
            obs_batch = convert_to_torch_tensor(obs_batch, self.device)

            # predict q-vals
            q_values, hiddens = mac(self.model, obs_batch, state_batches)
            avail_actions = convert_to_torch_tensor(action_mask, self.device)
            masked_q_values = q_values.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")
            masked_q_values_folded = torch.reshape(masked_q_values, [-1] + list(masked_q_values.shape)[2:])

            # select action
            action_dist = self.dist_class(masked_q_values_folded, self.model)
            actions, logp = self.exploration.get_exploration_action(
                action_distribution=action_dist,
                timestep=timestep,
                explore=explore,
            )

            actions = actions.cpu().numpy()
            hiddens = [s.view(self.n_agents, -1).cpu().numpy() for s in hiddens]

            # store q values selected in this time step for callbacks
            q_values = to_numpy(masked_q_values.squeeze()).tolist()

            results = convert_to_non_torch_type((actions, hiddens, {'q-values': [q_values]}))

        return results

    def compute_single_action(self, *args, **kwargs) -> \
            Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        return super().compute_single_action(*args, **kwargs)

    @override(Policy)
    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[
                "Policy", SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        # observation sharing
        if len(other_agent_batches) > 0:
            sample_batch = sample_batch.copy()
            n_obs = []
            n_next_obs = []
            for n_id, (policy, batch) in other_agent_batches.items():
                n_obs.append(
                    policy.get_message(batch[SampleBatch.OBS])
                )
                n_next_obs.append(
                    policy.get_message(batch[SampleBatch.NEXT_OBS])
                )
            sample_batch[NEIGHBOUR_OBS] = np.array(n_obs).reshape(
                len(sample_batch), len(other_agent_batches), -1)
            sample_batch[NEIGHBOUR_NEXT_OBS] = np.array(n_next_obs).reshape(
                len(sample_batch), len(other_agent_batches), -1)
        return sample_batch

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        # Set Model to train mode.
        if self.model:
            self.model.train()
            self.auxiliary_model.train()
            if self.use_comm:
                self.comm_net.train()
                self.comm_agg.train()

        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self,
            train_batch=samples,
            result=learn_stats,
        )

        (action_mask, actions, env_global_state, mask, next_action_mask, next_env_global_state,
         next_obs, obs, rewards, terminated, n_obs, n_next_obs, seq_lens) = preprocess_trajectory_batch(
            policy=self,
            samples=samples,
            has_neighbour_data=NEIGHBOUR_OBS in samples and NEIGHBOUR_NEXT_OBS in samples,
        )

        loss_out, mask, masked_td_error, chosen_action_qvals, targets = self.compute_trajectory_q_loss(
            rewards,
            actions,
            terminated,
            mask,
            obs,
            next_obs,
            action_mask,
            next_action_mask,
            env_global_state,
            next_env_global_state,
            n_obs,
            n_next_obs,
        )

        # Optimise
        self.optimiser.zero_grad()
        loss_out.backward()
        grad_norm_clipping_ = self.config["grad_clip"]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, grad_norm_clipping_)
        self.optimiser.step()

        mask_elems = mask.sum().item()
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        data = {
            LEARNER_STATS_KEY: stats,
            "model": self.model.metrics(),
            "custom_metrics": learn_stats,
            "seq_lens": seq_lens,
        }
        data.update(self.model.tower_stats)
        return data

    @override(Policy)
    def get_initial_state(self):
        return [
            s.expand([self.n_agents, -1]).cpu().numpy()
            for s in self.model.get_initial_state()
        ]

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.auxiliary_model.load_state_dict(self._device_dict(weights["auxiliary_model"]))
        self.auxiliary_model_target.load_state_dict(self._device_dict(weights["auxiliary_model_target"]))

        if self.use_comm and "comm_net" in weights:
            self.comm_net.load_state_dict(self._device_dict(weights["comm_net"]))
            self.comm_net_target.load_state_dict(self._device_dict(weights["comm_net_target"]))
            self.comm_agg.load_state_dict(self._device_dict(weights["comm_agg"]))
            self.comm_agg_target.load_state_dict(self._device_dict(weights["comm_agg_target"]))

    @override(Policy)
    def get_weights(self):
        wts = {
            "model": self._cpu_dict(self.model.state_dict()),
            "auxiliary_model": self._cpu_dict(self.auxiliary_model.state_dict()),
            "auxiliary_model_target": self._cpu_dict(self.auxiliary_model_target.state_dict()),
        }

        if self.use_comm:
            wts["comm_net"] = self._cpu_dict(self.comm_net.state_dict())
            wts["comm_net_target"] = self._cpu_dict(self.comm_net_target.state_dict())
            wts["comm_agg"] = self._cpu_dict(self.comm_agg.state_dict())
            wts["comm_agg_target"] = self._cpu_dict(self.comm_agg_target.state_dict())
        return wts

    @override(Policy)
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: to_numpy(v) for k, v in state_dict.items()}

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()
        }

    def update_target(self):
        self.auxiliary_model_target = soft_update(self.auxiliary_model_target, self.auxiliary_model, self.tau)
        if self.use_comm:
            self.comm_net_target = soft_update(self.comm_net_target, self.comm_net, self.tau)
            self.comm_agg_target = soft_update(self.comm_agg_target, self.comm_agg, self.tau)

    def convert_batch_to_tensor(self, data_dict):
        obs_batch = SampleBatch(data_dict)
        obs_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )
        return obs_batch

    def add_comm_msg(self, model, ob, next_ob, B, T, n_obs, n_next_obs):
        local_msg = model(ob).unsqueeze(2)
        msgs = torch.cat([local_msg, n_obs], dim=2)
        agg_msg = self.aggregate_messages(msgs.view(B * T, *msgs.shape[2:])).view(B, T, -1)
        ob = torch.cat([ob, agg_msg], dim=-1)

        next_local_msg = model(next_ob).unsqueeze(2)
        next_msgs = torch.cat([next_local_msg, n_next_obs], dim=2)
        agg_next_msg = self.aggregate_messages(
            next_msgs.view(B * T, *msgs.shape[2:]), True
        ).view(B, T, -1)
        next_ob = torch.cat([next_ob, agg_next_msg], dim=-1)

        return ob, next_ob

    def compute_trajectory_q_loss(self,
                                  rewards,
                                  actions,
                                  terminated,
                                  mask,
                                  obs,
                                  next_obs,
                                  action_mask,
                                  next_action_mask,
                                  state=None,
                                  next_state=None,
                                  neighbour_obs=None,
                                  neighbour_next_obs=None):
        """
        Computes the Q loss.

        Args:
            rewards: Tensor of shape [B, T]
            actions: Tensor of shape [B, T]
            terminated: Tensor of shape [B, T]
            mask: Tensor of shape [B, T]
            obs: Tensor of shape [B, T, obs_size]
            next_obs: Tensor of shape [B, T, obs_size]
            action_mask: Tensor of shape [B, T, n_actions]
            next_action_mask: Tensor of shape [B, T, n_actions]
            state: Tensor of shape [B, T, state_dim] (optional)
            next_state: Tensor of shape [B, T, state_dim] (optional)
            neighbour_obs: Tensor of shape [B, T, num_neighbours, obs_size]
            neighbour_next_obs: Tensor of shape [B, T, num_neighbours, obs_size]
            lds_weights: Tensor of shape [B, T]
        """
        B, T = obs.shape[0], obs.shape[1]
        target_obs, raw_obs = obs, obs
        target_next_obs, raw_next_obs = next_obs, next_obs

        if self.use_comm:
            obs, next_obs = self.add_comm_msg(
                self.comm_net, obs, next_obs, B, T, neighbour_obs, neighbour_next_obs
            )
            target_obs, target_next_obs = self.add_comm_msg(
                self.comm_net_target, target_obs, target_next_obs, B, T, neighbour_obs, neighbour_next_obs
            )

        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)
        whole_obs = whole_obs.unsqueeze(2)
        target_whole_obs = torch.cat((target_obs[:, 0:1], target_next_obs), axis=1)
        target_whole_obs = target_whole_obs.unsqueeze(2)
        qe_bar_out, qe_bar_h = unroll_mac_squeeze_wrapper(unroll_mac(self.auxiliary_model_target, target_whole_obs))
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)

        # Auxiliary encoder objective
        # Qe(s, a_i)
        qe_out, qe_h = unroll_mac_squeeze_wrapper(unroll_mac(self.auxiliary_model, whole_obs))
        qe_qvals = torch.gather(qe_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Get LDS weights
        lds_qe_bar_out_qvals = qe_bar_out[:, 1:]
        lds_qe_bar_out_qvals[ignore_action_tp1] = -np.inf
        lds_qe_bar_out_qvals = lds_qe_bar_out_qvals.max(dim=2)[0]
        lds_targets = rewards + self.config["gamma"] * (1 - terminated) * lds_qe_bar_out_qvals
        lds_targets = lds_targets / torch.clamp(torch.max(lds_targets), 1e-6)
        targets_flat = to_numpy(lds_targets).reshape(-1, )
        lds_weights, bin_index_per_label = get_lds_weights(
            samples=SampleBatch({
                SampleBatch.REWARDS: targets_flat,
            }),
            **self.fds_config,
        )
        bin_index_per_label = convert_to_torch_tensor(bin_index_per_label, self.device)
        lds_weights = convert_to_torch_tensor(lds_weights, self.device).reshape(*lds_targets.shape)

        # Qi(s', a'_i*)
        qi_out, qi_h = unroll_mac_squeeze_wrapper(
            unroll_mac(self.model, whole_obs, lds_labels=bin_index_per_label, epoch=self._fds_epoch),
        )
        qi_out_sp = qi_out[:, 1:]
        # Mask out unavailable actions for the t+1 step
        qi_out_sp[ignore_action_tp1] = -np.inf
        qi_out_sp_qvals = qi_out_sp.max(dim=2)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.config["gamma"] * (1 - terminated) * qi_out_sp_qvals

        # Qe_i TD error
        qe_td_error = (qe_qvals - targets.detach())
        mask = mask.expand_as(qe_td_error)
        # 0-out the targets that came from padded data
        masked_td_error = qe_td_error * mask
        qe_loss = huber_loss(masked_td_error).sum() / mask.sum()
        self.model.tower_stats["Qe_loss"] = to_numpy(qe_loss)
        self.model.tower_stats["td_error"] = to_numpy(qe_td_error)

        # Qi function objective
        # Qi(s, a)
        qi_out_s_qvals = torch.gather(qi_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        # Qe_bar(s, a)
        # qe_bar_out = _unroll_mac(self.auxiliary_model_target, target_whole_obs).squeeze(2)
        qe_bar_out_qvals = torch.gather(qe_bar_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        qi_weights = torch.where(qe_bar_out_qvals > qi_out_s_qvals, 1.0, self.lamda)
        qi_loss = qi_weights * lds_weights * huber_loss(qi_out_s_qvals - qe_bar_out_qvals.detach())
        qi_loss = torch.sum(qi_loss * mask) / mask.sum()

        self.model.tower_stats["Qi_loss"] = to_numpy(qi_loss)

        # combine losses
        loss = qe_loss + qi_loss
        self.model.tower_stats["loss"] = to_numpy(loss)

        save_representations(
            obs=obs,
            latent_rep=qe_h[:, :-1],
            model_out=qe_qvals,
            target=targets,
            reward=rewards,
            filename_prefix=self.agent_id + "_",
        )

        return loss, mask, masked_td_error, qi_out_s_qvals, targets

    def get_message(self, obs):
        if obs.ndim < 2:
            obs = np.expand_dims(obs, axis=0)
        obs, _, _ = unpack_observation(
            self,
            obs,
        )
        if self.use_comm:
            obs = convert_to_torch_tensor(obs, self.device).float()
            obs = to_numpy(self.comm_net(obs))
        return obs

    def aggregate_messages(self, msg, is_target=False):
        # agg_msg = torch.sum(msg, dim=1, keepdim=True)
        agg_msg = self.comm_agg_target(msg) if is_target else self.comm_agg(msg)
        return agg_msg

    @torch.no_grad()
    def update_fds_running_stats(self, samples: SampleBatch):
        if hasattr(self.model, "FDS"):
            self.model.eval()
            if self.use_comm:
                self.comm_net.eval()
                self.comm_agg.eval()

            (action_mask, actions, env_global_state, mask, next_action_mask, next_env_global_state,
             next_obs, obs, rewards, terminated, n_obs, n_next_obs, seq_lens) = preprocess_trajectory_batch(
                policy=self,
                samples=samples,
                has_neighbour_data=NEIGHBOUR_OBS in samples and NEIGHBOUR_NEXT_OBS in samples,
            )

            B, T = obs.shape[0], obs.shape[1]
            target_obs, raw_obs = obs, obs
            target_next_obs, raw_next_obs = next_obs, next_obs

            if self.use_comm:
                obs, next_obs = self.add_comm_msg(
                    self.comm_net, obs, next_obs, B, T, n_obs, n_next_obs
                )
                target_obs, target_next_obs = self.add_comm_msg(
                    self.comm_net_target, target_obs, target_next_obs, B, T, n_obs, n_next_obs
                )

            # append the first element of obs + next_obs to get new one
            whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)
            whole_obs = whole_obs.unsqueeze(2)
            target_whole_obs = torch.cat((target_obs[:, 0:1], target_next_obs), axis=1)
            target_whole_obs = target_whole_obs.unsqueeze(2)
            qe_bar_out, qe_bar_h = unroll_mac_squeeze_wrapper(unroll_mac(self.auxiliary_model_target, target_whole_obs))
            ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
            # Qi(s', a'_i*)
            _, qi_h = unroll_mac_squeeze_wrapper(unroll_mac(self.model, whole_obs))

            # Get LDS weights
            lds_qe_bar_out_qvals = qe_bar_out[:, 1:]
            lds_qe_bar_out_qvals[ignore_action_tp1] = -np.inf
            lds_qe_bar_out_qvals = lds_qe_bar_out_qvals.max(dim=2)[0]
            lds_targets = rewards + self.config["gamma"] * (1 - terminated) * lds_qe_bar_out_qvals
            lds_targets = lds_targets / torch.clamp(torch.max(lds_targets), 1e-6)
            targets_flat = to_numpy(lds_targets).reshape(-1, )
            lds_weights, bin_index_per_label = get_lds_weights(
                samples=SampleBatch({
                    SampleBatch.REWARDS: targets_flat,
                }),
                **self.fds_config,
            )
            bin_index_per_label = convert_to_torch_tensor(bin_index_per_label, self.device)

            # stats update
            self.model.FDS.update_last_epoch_stats(self._fds_epoch)
            self.model.FDS.update_running_stats(qi_h[:, :-1].reshape(B * T, -1), bin_index_per_label, self._fds_epoch)
            self._fds_epoch += 1
