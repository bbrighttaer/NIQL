import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
from gym.spaces import Dict as GymDict
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils import override, PiecewiseSchedule
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

from niql.config import FINGERPRINT_SIZE
from niql.models import SimpleCommNet, AttentionCommMessagesAggregator
from niql.utils import get_size
from niql.utils import unpack_observation, preprocess_trajectory_batch, to_numpy, NEIGHBOUR_OBS, NEIGHBOUR_NEXT_OBS, \
    batch_message_inter_agent_sharing, mac

logger = logging.getLogger(__name__)


class NIQLBasePolicy(LearningRateSchedule, Policy, ABC):

    def __init__(self, obs_space, action_space, config, models_factory_class):
        self.framework = "torch"
        config = dict(DEFAULT_CONFIG, **config)

        Policy.__init__(self, obs_space, action_space, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

        self.n_agents = 1
        self.policy_id = config["policy_id"]
        self.lamda = config.get("lambda", 0.)
        self.tau = config["tau"]
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
        self.neighbour_messages = []

        self.tdw_schedule = PiecewiseSchedule(
            framework=None,
            endpoints=self.config["tdw_schedule"],
        )

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
        model_arch_args = config["model"]["custom_model_config"]["model_arch_args"]
        self.use_comm = model_arch_args.get("comm_dim", 0) > 0
        comm_dim = 0
        self.comm_net = None
        self.comm_net_target = None
        if self.use_comm:
            fp_size = config["comm_num_agents"]
            comm_dim = model_arch_args["comm_dim"] + fp_size
            comm_hdim = model_arch_args["comm_hdim"]
            config["model"]["comm_dim"] = comm_dim
            config["model"]["comm_aggregator_dim"] = model_arch_args["comm_aggregator_dim"]
            agent_index = int(self.policy_id.split("_")[-1])

            self.comm_net = SimpleCommNet(
                self.obs_size, comm_hdim, comm_dim, agent_index, fp_size, discrete=False
            ).to(self.device)
            self.comm_net_target = SimpleCommNet(
                self.obs_size, comm_hdim, comm_dim, agent_index, fp_size, discrete=False
            ).to(self.device)

        # create models
        self.params = []
        models_factory_class.__init__(self, agent_obs_space, action_space, config, core_arch)

        self.exploration = self._create_exploration()
        self.dist_class = TorchCategorical

        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0
        self._training_iteration_num = 0
        self._global_update_count = 0

        # create comm aggregators
        if self.use_comm:
            query_dim = model_arch_args["hidden_state_size"] if self._is_recurrent else self.obs_size
            self.comm_aggregator = AttentionCommMessagesAggregator(
                query_dim=query_dim,
                comm_dim=comm_dim,
                hidden_dims=model_arch_args["comm_aggregator_hdims"],
                output_dim=model_arch_args["comm_aggregator_dim"],
            ).to(self.device)
            self.comm_aggregator_target = AttentionCommMessagesAggregator(
                query_dim=query_dim,
                comm_dim=comm_dim,
                hidden_dims=model_arch_args["comm_aggregator_hdims"],
                output_dim=model_arch_args["comm_aggregator_dim"],
            ).to(self.device)

        # optimizer
        self.comm_params = []
        if self.use_comm:
            self.params += list(self.comm_net.parameters()) + list(self.comm_aggregator.parameters())
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

    @property
    def _optimizers(self):
        return [self.optimiser, self.comm_params]

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
        self.switch_models_to_eval_mode()

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
                    query = state_batches[0].unsqueeze(0) if self.is_recurrent else obs_batch
                    msg = self.aggregate_messages(False, query, msg)
                    self.neighbour_messages.clear()
                obs_batch = torch.cat([obs_batch, msg], dim=-1)

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
        # observation sharing
        sample_batch = batch_message_inter_agent_sharing(sample_batch, other_agent_batches)
        return sample_batch

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        # Set Model to train mode.
        self.switch_models_to_train_mode()

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
        if self.use_comm:
            comm_net_grad_norm = torch.nn.utils.clip_grad_norm_(self.comm_net.parameters(), grad_norm_clipping_)
            comm_agg_grad_norm = torch.nn.utils.clip_grad_norm_(self.comm_aggregator.parameters(), grad_norm_clipping_)
        else:
            comm_net_grad_norm = comm_agg_grad_norm = 0.
        self.optimiser.step()

        mask_elems = mask.sum().item()
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
            "comm_net_norm": comm_net_grad_norm if isinstance(comm_net_grad_norm, float) else comm_net_grad_norm.item(),
            "comm_agg_norm": comm_agg_grad_norm if isinstance(comm_agg_grad_norm, float) else comm_agg_grad_norm.item()
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
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: to_numpy(v) for k, v in state_dict.items()}

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()
        }

    def convert_batch_to_tensor(self, data_dict):
        obs_batch = SampleBatch(data_dict)
        obs_batch.set_get_interceptor(
            partial(convert_to_torch_tensor, device=self.device)
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

    def get_message(self, obs):
        if obs.ndim < 2:
            obs = np.expand_dims(obs, axis=0)
        obs, _, _ = unpack_observation(
            self,
            obs,
        )
        if self.use_comm:
            self.comm_net.eval()
            obs = convert_to_torch_tensor(obs, self.device).float()
            msg = self.comm_net(obs)
            obs = to_numpy(msg)
        return obs

    def aggregate_messages(self, is_target, query, all_messages):
        """
        Aggregates all messages of the communication module.

        :param query: query tensor of shape (batch_size, timesteps, obs_dim)
        :param all_messages: concatenated local and neighbour messages, shape (batch_size * timesteps, num_messages, comm_dim)
        :param is_target: whether the target model should be used.
        :return: tensor of shape (batch_size * timesteps, comm_dim + aggregator_output_dim)
        """
        # merge query batch and timestep dims
        B, T = query.shape[:2]
        query = query.view(B * T, -1)

        # select aggregator to be used
        model = self.comm_aggregator_target if is_target else self.comm_aggregator

        # get local message
        local_msg = all_messages[:, 0:1, :]

        # aggregate messages
        agg_msg = model(query, all_messages).unsqueeze(1)

        # construct final output
        out = torch.cat([local_msg, agg_msg], dim=-1)

        return out

    @abstractmethod
    def switch_models_to_eval_mode(self):
        ...

    @abstractmethod
    def switch_models_to_train_mode(self):
        ...

    @abstractmethod
    def set_weights(self, weights):
        ...

    @abstractmethod
    def get_weights(self):
        ...

    @abstractmethod
    def update_target(self):
        ...

    @abstractmethod
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
            neighbour_obs: Tensor of shape [B, T, num_neighbours, obs_size]
            neighbour_next_obs: Tensor of shape [B, T, num_neighbours, obs_size]
        """
        ...
