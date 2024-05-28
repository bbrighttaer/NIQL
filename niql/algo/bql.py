import functools
import logging
import random
import string
from collections import Counter
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
from gym.spaces import Dict as GymDict
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.qmix.qmix_policy import _mac
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import convert_to_torch_tensor, convert_to_non_torch_type, huber_loss
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

from niql.envs import DEBUG_ENVS
from niql.models import DRQNModel
from niql.utils import preprocess_trajectory_batch, unpack_observation, NEIGHBOUR_NEXT_OBS, NEIGHBOUR_OBS, unroll_mac, \
    unroll_mac_squeeze_wrapper, to_numpy, get_size, soft_update, tb_add_scalar, tb_add_scalars

logger = logging.getLogger(__name__)


class BQLPolicy(LearningRateSchedule, Policy):
    """
    Implementation of Best Possible Q-learning
    """

    def __init__(self, obs_space, action_space, config):
        self.framework = "torch"
        config = dict(DEFAULT_CONFIG, **config)
        Policy.__init__(self, obs_space, action_space, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        self.n_agents = 1
        # self.agent_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        self.policy_id = config["policy_id"]
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
        sample_batch["agent_ids"] = np.array([self.policy_id] * len(sample_batch))
        return sample_batch

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        # Set Model to train mode.
        if self.model:
            self.model.train()
            self.auxiliary_model.train()

        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self,
            train_batch=samples,
            result=learn_stats,
        )

        (action_mask, actions, env_global_state, mask, next_action_mask, next_env_global_state,
         next_obs, obs, rewards, weights, terminated, n_obs, n_next_obs, seq_lens) = preprocess_trajectory_batch(
            policy=self,
            samples=samples,
            has_neighbour_data=NEIGHBOUR_OBS in samples and NEIGHBOUR_NEXT_OBS in samples,
        )

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
            "qe_loss": self.model.tower_stats["Qe_loss"],
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

    @override(Policy)
    def get_weights(self):
        wts = {
            "model": self._cpu_dict(self.model.state_dict()),
            "auxiliary_model": self._cpu_dict(self.auxiliary_model.state_dict()),
            "auxiliary_model_target": self._cpu_dict(self.auxiliary_model_target.state_dict()),
        }
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
        self.auxiliary_model_target = soft_update(self.auxiliary_model_target, self.auxiliary_model, self.tau)

    def convert_batch_to_tensor(self, data_dict):
        obs_batch = SampleBatch(data_dict)
        obs_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )
        return obs_batch

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
                                  next_state=None,
                                  neighbour_obs=None,
                                  neighbour_next_obs=None):
        """
        Computes the Q loss.

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
            neighbour_obs: Tensor of shape [B, T, num_neighbours, obs_size]
            neighbour_next_obs: Tensor of shape [B, T, num_neighbours, obs_size]
        """
        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)
        whole_obs = whole_obs.unsqueeze(2)

        # Auxiliary encoder objective
        # Qe(s, a_i)
        qe_out, qe_h_out = unroll_mac_squeeze_wrapper(unroll_mac(self.auxiliary_model, whole_obs))
        qe_qvals = torch.gather(qe_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Qi(s', a'_i*)
        qi_out, qi_h_out = unroll_mac_squeeze_wrapper(unroll_mac(self.model, whole_obs))
        qi_out_sp = qi_out[:, 1:]
        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        qi_out_sp[ignore_action_tp1] = -np.inf
        qi_out_sp_qvals = qi_out_sp.max(dim=2)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.config["gamma"] * (1 - terminated) * qi_out_sp_qvals

        # Qe_i TD error
        qe_td_error = qe_qvals - targets.detach()
        mask = mask.expand_as(qe_td_error)
        # 0-out the targets that came from padded data
        masked_td_error = qe_td_error * mask
        qe_loss = huber_loss(masked_td_error).sum() / mask.sum()
        self.model.tower_stats["Qe_loss"] = to_numpy(qe_loss)
        self.model.tower_stats["td_error"] = to_numpy(qe_td_error)
        tb_add_scalar(self, "qe_loss", qe_loss.item())

        # gather td error for each unique target for analysis (matrix game case - discrete reward)
        if self.config.get("env_name") in DEBUG_ENVS:
            unique_targets = torch.unique(targets.int())
            mean_td_stats = {
               t.item(): torch.mean(torch.abs(masked_td_error).view(-1,)[targets.view(-1,).int() == t]).item()
                for t in unique_targets
            }
            tb_add_scalars(self, "td-error_dist", mean_td_stats)

        # Qi function objective
        # Qi(s, a)
        qi_out_s_qvals = torch.gather(qi_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        # Qe_bar(s, a)
        qe_bar_out, qe_bar_h_out = unroll_mac_squeeze_wrapper(unroll_mac(self.auxiliary_model_target, whole_obs))
        qe_bar_out_qvals = torch.gather(qe_bar_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        qi_weights = torch.where(qe_bar_out_qvals > qi_out_s_qvals, 1.0, self.lamda)
        qi_loss = qi_weights * huber_loss(qi_out_s_qvals - qe_bar_out_qvals.detach())
        qi_loss = torch.sum(qi_loss * mask) / mask.sum()
        self.model.tower_stats["Qi_loss"] = to_numpy(qi_loss)
        tb_add_scalar(self, "qi_loss", qi_loss.item())

        # combine losses
        loss = qe_loss + qi_loss
        self.model.tower_stats["loss"] = to_numpy(loss)
        tb_add_scalar(self, "loss", loss.item())

        # save_representations(
        #     obs=obs,
        #     latent_rep=qe_h_out[:, :-1],
        #     model_out=qe_qvals,
        #     target=targets,
        #     reward=rewards,
        # )

        return loss, mask, masked_td_error, qi_out_s_qvals, targets
