import functools
import logging
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Dict as GymDict
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.qmix.qmix_policy import _unroll_mac, _mac
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import convert_to_torch_tensor, convert_to_non_torch_type, huber_loss
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

from niql import distance_metrics
from niql.models import DRQNModel, MultiHeadSelfAttentionEncoder, FCNEncoder
from niql.utils import preprocess_trajectory_batch, unpack_observation, NEIGHBOUR_NEXT_OBS, NEIGHBOUR_OBS

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


class BQLPolicy(Policy):
    """
    Implementation of Best Possible Q-learning
    """

    def __init__(self, obs_space, action_space, config):
        self.framework = "torch"
        config = dict(DEFAULT_CONFIG, **config)
        super().__init__(obs_space, action_space, config)
        self.n_agents = 1
        config["model"]["n_agents"] = self.n_agents
        self.lamda = config["lambda"]
        self.tau = config["tau"]
        self.n_actions = action_space.n
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        config["model"]["full_obs_space"] = obs_space
        core_arch = config["model"]["custom_model_config"]["model_arch_args"]["core_arch"]
        self.reward_standardize = config["reward_standardize"]
        self.shared_neighbour_obs = []

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
        if self.config["use_obs_encoder"]:
            self.obs_encoder = FCNEncoder(
                input_dim=self.obs_size,
                num_heads=config["model"]["custom_model_config"]["model_arch_args"]["mha_num_heads"],
                device=self.device,
            ).to(self.device)
        else:
            self.obs_encoder = None
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
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

        self.auxiliary_target_model = ModelCatalog.get_model_v2(
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
        if self.obs_encoder:
            self.params += list(self.obs_encoder.parameters())

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
            if self.obs_encoder:
                self.obs_encoder.eval()

        with torch.no_grad():
            obs_batch = convert_to_torch_tensor(obs_batch, self.device)
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]

            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)

            if self.config["use_obs_encoder"]:
                if self.shared_neighbour_obs:
                    n = len(self.shared_neighbour_obs)
                    obs_batch = obs_batch.reshape(1, 1, -1)
                    neighbour_obs = convert_to_torch_tensor(
                        np.array(self.shared_neighbour_obs).reshape(1, n, -1), self.device)
                    obs_batch = torch.cat([obs_batch, neighbour_obs], axis=1)
                    self.shared_neighbour_obs.clear()
                obs_batch = convert_to_torch_tensor(obs_batch, self.device)
                obs_batch, _ = self.obs_encoder(obs_batch)
                obs_batch = obs_batch.unsqueeze(1)
            else:
                obs_batch = convert_to_torch_tensor(obs_batch, self.device)

            # predict q-vals
            q_values, hiddens = _mac(self.model, obs_batch, state_batches)
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
            q_values = masked_q_values.squeeze().cpu().detach().numpy().tolist()

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
            for n_id, (_, batch) in other_agent_batches.items():
                n_obs.append(
                    batch[SampleBatch.OBS]
                )
                n_next_obs.append(
                    batch[SampleBatch.NEXT_OBS]
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
            if self.obs_encoder:
                self.obs_encoder.train()

        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self,
            train_batch=samples,
            result=learn_stats,
        )

        (action_mask, actions, env_global_state, mask, next_action_mask, next_env_global_state,
         next_obs, obs, rewards, terminated, n_obs, n_next_obs) = preprocess_trajectory_batch(
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
            "custom_metrics": learn_stats
        }
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
        self.auxiliary_target_model.load_state_dict(self._device_dict(weights["auxiliary_target_model"]))
        if "obs_encoder" in weights and self.obs_encoder:
            self.obs_encoder.load_state_dict(self._device_dict(weights["obs_encoder"]))

    @override(Policy)
    def get_weights(self):
        wts = {
            "model": self._cpu_dict(self.model.state_dict()),
            "auxiliary_model": self._cpu_dict(self.auxiliary_model.state_dict()),
            "auxiliary_target_model": self._cpu_dict(self.auxiliary_target_model.state_dict()),
        }
        if self.obs_encoder:
            wts["obs_encoder"] = self._cpu_dict(self.obs_encoder.state_dict())
        return wts

    @override(Policy)
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()
        }

    def update_target(self):
        self.auxiliary_target_model = soft_update(self.auxiliary_target_model, self.auxiliary_model, self.tau)

    def compute_q_losses(self, train_batch: SampleBatch) -> TensorType:
        obs = train_batch[SampleBatch.OBS]
        actions = train_batch[SampleBatch.ACTIONS]
        batch_rewards = train_batch[SampleBatch.REWARDS]
        next_obs = train_batch[SampleBatch.NEXT_OBS]
        convert = True

        # ----------------------- Reward reconciliation ------------------
        if self.config["reconcile_rewards"]:
            if self.config["use_obs_encoder"]:
                def projection(x, n_x):
                    x = convert_to_torch_tensor(x, self.device).unsqueeze(1)
                    n_obs = convert_to_torch_tensor(n_x, self.device)
                    x = torch.cat([x, n_obs], dim=1)
                    x = self.obs_encoder(x)
                    return x
                obs = projection(obs, train_batch[NEIGHBOUR_OBS])
                next_obs = projection(next_obs, train_batch[NEIGHBOUR_NEXT_OBS])
                convert = False
                batch_rewards = distance_metrics.batch_cosine_similarity_reward_update_np(
                    obs=obs.cpu().detach().numpy(),
                    actions=actions,
                    rewards=batch_rewards,
                    threshold=0.99,
                )
                batch_rewards = convert_to_torch_tensor(batch_rewards, self.device)
                obs = {SampleBatch.OBS: obs}
                next_obs = {SampleBatch.OBS: next_obs}
            else:
                batch_rewards = distance_metrics.batch_cosine_similarity_reward_update_np(
                    obs=obs,
                    actions=actions,
                    rewards=batch_rewards,
                )

        # batch preprocessing ops
        if convert:
            obs = self.convert_batch_to_tensor({
                SampleBatch.OBS: convert_to_torch_tensor(obs, self.device)
            })
            next_obs = self.convert_batch_to_tensor({
                SampleBatch.OBS: convert_to_torch_tensor(next_obs, self.device)
            })
            batch_rewards = convert_to_torch_tensor(batch_rewards, self.device)

        train_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )

        # get hidden states for RNN case
        i = 0
        state_batches_h = []
        while "state_in_{}".format(i) in train_batch:
            state_batches_h.append(train_batch["state_in_{}".format(i)])
            i += 1
        if self._is_recurrent:
            assert state_batches_h
        i = 0
        state_batches_h_prime = []
        while "state_out_{}".format(i) in train_batch:
            state_batches_h_prime.append(train_batch["state_out_{}".format(i)])
            i += 1
        if self._is_recurrent:
            assert state_batches_h_prime
        seq_lens = train_batch.get(SampleBatch.SEQ_LENS)

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), num_classes=self.action_space.n)
        one_hot_selection = one_hot_selection.to(self.device)

        # ----------------------- Auxiliary model objective --------------
        # compute q-vals
        q_e, _ = self.auxiliary_model(obs, state_batches_h, seq_lens)
        q_e = torch.sum(q_e * one_hot_selection, dim=1)

        # compute estimate of the best possible value starting from state at t + 1
        q, _ = self.model(next_obs, state_batches_h_prime, seq_lens)
        dones = train_batch[SampleBatch.DONES].float()
        q_best_one_hot_selection = F.one_hot(torch.argmax(q, 1), self.action_space.n)
        q_best = torch.sum(q * q_best_one_hot_selection, 1)
        q_best = (1.0 - dones) * q_best

        # compute RHS of bellman equation
        q_e_target = batch_rewards + self.config["gamma"] * q_best

        # Compute the error (Square/Huber).
        td_error_q_e = q_e - q_e_target.detach()
        loss_qe = torch.mean(huber_loss(td_error_q_e))
        self.model.tower_stats["loss_qe"] = loss_qe
        self.model.tower_stats["td_error"] = td_error_q_e

        # --------------------------- Main model -------------------------
        qt, _ = self.model(obs, state_batches_h, seq_lens)
        qt_selected = torch.sum(qt * one_hot_selection, dim=1)
        q_bar_e, _ = self.auxiliary_target_model(obs, state_batches_h, seq_lens)
        q_bar_e_selected = torch.sum(q_bar_e * one_hot_selection, dim=1)
        qt_weight = torch.where(q_bar_e_selected > qt_selected, 1.0, self.lamda)
        # loss = weight * (qt_selected - q_bar_e_selected.detach()) ** 2
        loss = qt_weight * huber_loss(qt_selected - q_bar_e_selected.detach())
        loss = torch.mean(loss)

        self.model.tower_stats["loss"] = loss

        return loss + loss_qe, td_error_q_e.abs().sum().item(), qt_selected.mean().item(), q_e_target.mean().item()

    def convert_batch_to_tensor(self, data_dict):
        obs_batch = SampleBatch(data_dict)
        obs_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )
        return obs_batch

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
        """
        B, T = obs.shape[0], obs.shape[1]

        # reward reconciliation
        if self.config["reconcile_rewards"]:
            threshold = self.config["similarity_threshold"]
            if self.config.get("use_obs_encoder", False) and neighbour_obs is not None and neighbour_next_obs is not None:
                def projection(x, n_x):
                    x = convert_to_torch_tensor(x, self.device).unsqueeze(2)
                    n_obs = convert_to_torch_tensor(n_x, self.device)
                    x = torch.cat([x, n_obs], dim=2)
                    x, enc = self.obs_encoder(x.view(B * T, *x.shape[2:]))
                    x = x.view(B, T, -1)
                    return x, enc

                obs, encoding = projection(obs, neighbour_obs)
                next_obs, _ = projection(next_obs, neighbour_next_obs)
                batch_rewards = distance_metrics.batch_cosine_similarity_reward_update_torch(
                    obs=encoding.clone().detach().reshape(B * T, -1),
                    actions=actions.reshape(-1, 1),
                    rewards=rewards.reshape(-1, 1),
                    threshold=threshold,
                )
                rewards = batch_rewards.view(*rewards.shape)
            else:
                batch_rewards = distance_metrics.batch_cosine_similarity_reward_update_torch(
                    obs=obs.clone().detach().reshape(B * T, -1),
                    actions=actions.reshape(-1, 1),
                    rewards=rewards.reshape(-1, 1),
                    threshold=threshold,
                )
                rewards = batch_rewards.view(*rewards.shape)

        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)
        whole_obs = whole_obs.unsqueeze(2)

        # Auxiliary model objective
        # Qe(s, a_i)
        qe_out = _unroll_mac(self.auxiliary_model, whole_obs).squeeze(2)
        qe_qvals = torch.gather(qe_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Qi(s', a'_i*)
        qi_out = _unroll_mac(self.model, whole_obs).squeeze(2)
        qi_out_sp = qi_out[:, 1:]
        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
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
        self.model.tower_stats["Qe_loss"] = qe_loss
        self.model.tower_stats["td_error"] = qe_td_error

        # Qi function objective
        # Qi(s, a)
        qi_out_s_qvals = torch.gather(qi_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        # Qe_bar(s, a)
        qe_bar_out = _unroll_mac(self.auxiliary_target_model, whole_obs).squeeze(2)
        qe_bar_out_qvals = torch.gather(qe_bar_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        qi_weights = torch.where(qe_bar_out_qvals > qi_out_s_qvals, 1.0, self.lamda)
        qi_loss = qi_weights * huber_loss(qi_out_s_qvals - qe_bar_out_qvals.detach())
        qi_loss = torch.sum(qi_loss * mask) / mask.sum()
        self.model.tower_stats["Qi_loss"] = qi_loss

        # combine losses
        loss = qe_loss + qi_loss
        self.model.tower_stats["loss"] = loss

        return loss, mask, masked_td_error, qi_out_s_qvals, targets


class EncodingLoss(torch.nn.Module):
    def __init__(self, margin=1.0, threshold=0.01):
        super(EncodingLoss, self).__init__()
        self.margin = margin
        self.threshold = threshold

    def forward(self, obs, encoding):
        device = obs.device

        # index of each observation
        obs_idx = torch.arange(obs.shape[0]).to(device)

        # similarity among observations
        xx, yy = torch.meshgrid(obs_idx, obs_idx)
        obs_dist = F.pairwise_distance(obs[xx.ravel().to(device)], obs[yy.ravel().to(device)], keepdim=True)

        # similarity among encoding
        enc_dist = F.pairwise_distance(encoding[xx.ravel().to(device)], encoding[yy.ravel().to(device)], keepdim=True)

        # determine positive and negative samples among encoding
        distance_negative_mask = obs_dist > self.threshold
        distance_positive_mask = ~distance_negative_mask

        # Compute loss
        p_dist = enc_dist * distance_positive_mask
        n_dist = enc_dist * distance_negative_mask
        loss = torch.clamp(p_dist - n_dist + self.margin, min=0.0)
        loss = torch.mean(loss)

        return loss
