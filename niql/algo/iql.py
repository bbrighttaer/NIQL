import functools
import logging
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Dict as GymDict
from marllib.marl import JointQRNN
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.qmix.qmix_policy import _mac, _unroll_mac
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.utils.torch_ops import huber_loss
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

from niql.config import FINGERPRINT_SIZE
from niql.envs import DEBUG_ENVS
from niql.utils import unpack_observation, get_size, preprocess_trajectory_batch, to_numpy, tb_add_scalar, \
    tb_add_scalars

logger = logging.getLogger(__name__)


class IQLPolicy(Policy):

    def __init__(self, obs_space, action_space, config):
        self.framework = "torch"
        config = dict(DEFAULT_CONFIG, **config)
        super().__init__(obs_space, action_space, config)
        self.n_agents = 1
        self.policy_id = config["policy_id"]
        config["model"]["n_agents"] = self.n_agents
        self.use_fingerprint = config.get("use_fingerprint", False)
        self.info_sharing = config.get("info_sharing", False)
        self.n_actions = action_space.n
        self.h_size = config["model"]["lstm_cell_size"]
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        core_arch = config["model"]["custom_model_config"]["model_arch_args"]["core_arch"]
        self.reward_standardize = config["reward_standardize"]

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
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else JointQRNN
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else JointQRNN
        ).to(self.device)

        self.exploration = self._create_exploration()
        self.dist_class = TorchCategorical

        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0
        self._training_iteration_num = 0
        self._global_update_count = 0

        # optimizer
        self.params = self.model.parameters()
        if config["optimizer"] == "rmsprop":
            from torch.optim import RMSprop
            self.optimiser = RMSprop(
                params=self.params,
                lr=config["lr"])

        elif config["optimizer"] == "adam":
            from torch.optim import Adam
            self.optimiser = Adam(
                params=self.params,
                lr=config["lr"], )

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

        if self.use_fingerprint:
            obs_batch = self._pad_observation(obs_batch)

        # Switch to eval mode.
        if self.model:
            self.model.eval()

        with torch.no_grad():
            obs_batch = convert_to_torch_tensor(obs_batch, self.device)
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]

            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)

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
            q_values = to_numpy(masked_q_values.squeeze()).tolist()

            results = (actions, hiddens, {'q-values': [q_values]})

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
        if self.use_fingerprint:
            sample_batch = self._pad_sample_batch(sample_batch)
        return sample_batch

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        # Set Model to train mode.
        if self.model:
            self.model.train()

        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self,
            train_batch=samples,
            result=learn_stats,
        )

        (action_mask, actions, env_global_state, mask, next_action_mask, next_env_global_state,
         next_obs, obs, rewards, weights, terminated, _, _, seq_lens) = preprocess_trajectory_batch(self, samples)

        loss_out, mask, masked_td_error, chosen_action_qvals, targets = self.compute_trajectory_q_loss(
            rewards,
            weights,
            actions,
            terminated,
            mask,
            obs,
            next_obs,
            action_mask,
            next_action_mask,
            env_global_state,
            next_env_global_state,
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
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
        }

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
        self.target_model.load_state_dict(self.model.state_dict())
        logger.debug("Updated target networks")

    def convert_batch_to_tensor(self, data_dict):
        obs_batch = SampleBatch(data_dict)
        obs_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )
        return obs_batch

    def _pad_observation(self, obs_batch):
        """
        Add training iteration number and exploration value to observation.
        obs_batch is structured as [batch_size, feature_dim, ...]
        """
        b = obs_batch.shape[0]
        fp = np.array(
            [
                self.exploration.get_state()['cur_epsilon'],
                self._training_iteration_num
            ]
        ).reshape(1, -1)
        fp = np.tile(fp, (b, 1))
        obs_batch = np.concatenate([obs_batch[:, :-FINGERPRINT_SIZE], fp], axis=1)
        return obs_batch

    def _pad_sample_batch(self, sample_batch: SampleBatch):
        sample_batch = sample_batch.copy()
        obs = sample_batch[SampleBatch.OBS]
        obs = self._pad_observation(obs)
        next_obs = sample_batch[SampleBatch.NEXT_OBS]
        next_obs = self._pad_observation(next_obs)
        sample_batch[SampleBatch.OBS] = obs
        sample_batch[SampleBatch.NEXT_OBS] = next_obs
        return sample_batch

    def compute_q_losses(self, train_batch: SampleBatch) -> TensorType:
        # batch preprocessing ops
        obs = self.convert_batch_to_tensor({
            SampleBatch.OBS: train_batch[SampleBatch.OBS]
        })
        next_obs = self.convert_batch_to_tensor({
            SampleBatch.OBS: train_batch[SampleBatch.NEXT_OBS]
        })
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
        # i = 0
        # state_batches_h_prime = []
        # while "state_out_{}".format(i) in train_batch:
        #     state_batches_h_prime.append(train_batch["state_out_{}".format(i)])
        #     i += 1
        # assert state_batches_h_prime
        seq_lens = train_batch.get(SampleBatch.SEQ_LENS)

        # compute q-vals
        qt, _ = self.model(obs, state_batches_h, seq_lens)
        qt_prime, _ = self.target_model(next_obs, state_batches_h, seq_lens)

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), num_classes=self.action_space.n)
        qt_selected = torch.sum(qt * one_hot_selection, dim=1)

        # compute estimate of the best possible value starting from state at t + 1
        dones = train_batch[SampleBatch.DONES].float()
        qt_prime_best_one_hot_selection = F.one_hot(torch.argmax(qt_prime, 1), self.action_space.n)
        qt_prime_best = torch.sum(qt_prime * qt_prime_best_one_hot_selection, 1)
        qt_prime_best_masked = (1.0 - dones) * qt_prime_best

        # compute RHS of bellman equation
        qt_target = train_batch[SampleBatch.REWARDS] + self.config["gamma"] * qt_prime_best_masked

        # Compute the error (Square/Huber).
        td_error = qt_selected - qt_target.detach()
        loss = torch.mean(huber_loss(td_error))

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        self.model.tower_stats["loss"] = loss
        # TD-error tensor in final stats
        # will be concatenated and retrieved for each individual batch item.
        self.model.tower_stats["td_error"] = td_error

        return loss, td_error.abs().sum().item(), qt_selected.mean().item(), qt_target.mean().item()

    def compute_trajectory_q_loss(self,
                                  rewards,
                                  weights,
                                  actions,
                                  terminated,
                                  mask,
                                  obs,
                                  next_obs,
                                  action_mask,
                                  next_action_mask,
                                  state=None,
                                  next_state=None):
        """
        Computes the Q loss.
        Based on the JointQLoss of Marllib.

        Args:
            rewards: Tensor of shape [B, T]
            weights: Tensor of shape [B, T]
            actions: Tensor of shape [B, T]
            terminated: Tensor of shape [B, T]
            mask: Tensor of shape [B, T]
            obs: Tensor of shape [B, T, obs_size]
            next_obs: Tensor of shape [B, T, obs_size]
            action_mask: Tensor of shape [B, T, n_actions]
            next_action_mask: Tensor of shape [B, T, n_actions]
            state: Tensor of shape [B, T, state_dim] (optional)
            next_state: Tensor of shape [B, T, state_dim] (optional)
        """
        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)
        whole_obs = whole_obs.unsqueeze(2)

        # Calculate estimated Q-Values
        mac_out = _unroll_mac(self.model, whole_obs)
        mac_out = mac_out.squeeze(2)  # remove agent dimension

        # Pick the Q-Values for the actions taken -> [B * n_agents, T]
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Calculate the Q-Values necessary for the target
        target_mac_out = _unroll_mac(self.target_model, whole_obs)
        target_mac_out = target_mac_out.squeeze(2)  # remove agent dimension

        # we only need target_mac_out for raw next_obs part
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        # Max over target Q-Values
        if self.config["double_q"]:
            mac_out_tp1 = mac_out.clone().detach()
            mac_out_tp1 = mac_out_tp1[:, 1:]

            # mask out unallowed actions
            mac_out_tp1[ignore_action_tp1] = -np.inf

            # obtain best actions at t+1 according to policy NN
            cur_max_actions = mac_out_tp1.argmax(dim=2, keepdim=True)

            # use the target network to estimate the Q-values of policy
            # network's selected actions
            target_max_qvals = torch.gather(target_mac_out, 2, cur_max_actions).squeeze(2)
        else:
            target_max_qvals = target_mac_out.max(dim=2)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.config["gamma"] * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = weights * (chosen_action_qvals - targets.detach())
        self.model.tower_stats["td_error"] = to_numpy(td_error)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = huber_loss(masked_td_error).sum() / mask.sum()
        self.model.tower_stats["loss"] = to_numpy(loss)
        tb_add_scalar(self, "loss", loss.item())

        # gather td error for each unique target for analysis (matrix game case - discrete reward)
        if self.config.get("env_name") in DEBUG_ENVS:
            unique_targets = torch.unique(targets.int())
            mean_td_stats = {
                t.item(): torch.mean(torch.abs(masked_td_error).view(-1, )[targets.view(-1, ).int() == t]).item()
                for t in unique_targets
            }
            tb_add_scalars(self, "td-error_dist", mean_td_stats)

        return loss, mask, masked_td_error, chosen_action_qvals, targets
