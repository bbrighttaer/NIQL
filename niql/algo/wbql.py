# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import tree  # pip install dm_tree
from gym.spaces import Dict as Gym_Dict
from ray.rllib.agents.qmix.model import RNNModel, _get_size
from ray.rllib.agents.qmix.qmix_policy import _mac, _validate
from ray.rllib.execution.replay_buffer import *
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils import PiecewiseSchedule
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY

from niql.models import JointQRNN, JointQMLP
from niql.utils import _iql_unroll_mac, soft_update, target_distribution_weighting


# original _unroll_mac for next observation is different from Pymarl.
# thus we provide a new JointQLoss here
class JointQLoss(nn.Module):
    def __init__(self,
                 models,
                 aux_models,
                 aux_target_models,
                 n_agents,
                 n_actions,
                 tdw_schedule,
                 device,
                 double_q=True,
                 gamma=0.99,
                 lambda_=0.6):
        nn.Module.__init__(self)
        self.models = models
        self.aux_models = aux_models
        self.aux_target_models = aux_target_models
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = device
        self.tdw_schedule = PiecewiseSchedule(
            framework="torch",
            endpoints=tdw_schedule,
            outside_value=tdw_schedule[-1][-1]  # use value of last schedule
        )

    def forward(self,
                timestep,
                rewards,
                actions,
                terminated,
                mask,
                obs,
                next_obs,
                action_mask,
                next_action_mask,
                state=None,
                next_state=None):
        """Forward pass of the loss.

        Args:
            timestep: current timestep
            rewards: Tensor of shape [B, T, n_agents]
            actions: Tensor of shape [B, T, n_agents]
            terminated: Tensor of shape [B, T, n_agents]
            mask: Tensor of shape [B, T, n_agents]
            obs: Tensor of shape [B, T, n_agents, obs_size]
            next_obs: Tensor of shape [B, T, n_agents, obs_size]
            action_mask: Tensor of shape [B, T, n_agents, n_actions]
            next_action_mask: Tensor of shape [B, T, n_agents, n_actions]
            state: Tensor of shape [B, T, state_dim] (optional)
            next_state: Tensor of shape [B, T, state_dim] (optional)
        """
        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)

        # Calculate estimated Q-Values for all models
        def forward_prop(agent_models):
            mac_out = [
                _iql_unroll_mac(
                    agent_models[f"agent_{i}"], whole_obs[:, :, i:i + 1]
                ) for i in range(self.n_agents)
            ]
            mac_out = torch.cat(mac_out, dim=2)
            return mac_out

        qi_mac_out = forward_prop(self.models)
        qe_mac_out = forward_prop(self.aux_models)
        qe_target_mac_out = forward_prop(self.aux_target_models)

        # -------- Qe objective ------- #
        # Pick the q-values of chosen action estimated by Qe
        qe_chosen_action_qvals = torch.gather(
            qe_mac_out[:, :-1], dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        # Compute targets using Qi
        qi_mac_out_cloned = qi_mac_out.clone().detach()
        qi_targets = qi_mac_out_cloned[:, 1:]
        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        qi_targets[ignore_action_tp1] = -np.inf

        # Max over target Q-Values
        if self.double_q:
            # large bugs here in original QMixloss, the gradient is calculated
            # we fix this follow pymarl
            mac_out_tp1 = qi_mac_out.clone().detach()
            mac_out_tp1 = mac_out_tp1[:, 1:]

            # mask out unallowed actions
            mac_out_tp1[ignore_action_tp1] = -np.inf

            # obtain best actions at t+1 according to policy NN
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)

            # use the target network to estimate the Q-values of policy
            # network's selected actions
            target_max_qvals = torch.gather(qi_targets, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = qi_targets.max(dim=3)[0]

        assert target_max_qvals.min().item() != -np.inf, \
            "target_max_qvals contains a masked action; \
            there may be a state with no valid actions."

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = targets.detach() - qe_chosen_action_qvals

        # --------- Qi objective --------- #
        qi_chosen_action_qvals = torch.gather(
            qi_mac_out[:, :-1], dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        qe_target_chosen_action_qvals = torch.gather(
            qe_target_mac_out[:, :-1], dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        qi_error = qe_target_chosen_action_qvals.detach() - qi_chosen_action_qvals

        # determine hysteretic weights
        weights = torch.where(qi_error > 0, 1., self.lambda_)

        # Get target distribution weights
        if random.random() < self.tdw_schedule.value(timestep):
            tdw_weights = [
                target_distribution_weighting(
                    targets[:, :, i:i + 1], self.device, mask[:, :, i:i + 1]
                ) for i in range(self.n_agents)
            ]
            tdw_weights = torch.cat(tdw_weights, dim=2)
        else:
            tdw_weights = torch.ones_like(targets)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        qe_masked_td_error = td_error * mask
        qi_masked_error = qi_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (tdw_weights * (qe_masked_td_error ** 2)).sum() / mask.sum()
        loss += (weights * (qi_masked_error ** 2)).sum() / mask.sum()
        return loss, mask, qe_masked_td_error, qi_chosen_action_qvals, targets


class WBQLPolicy(LearningRateSchedule, Policy):

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qmix.qmix.DEFAULT_CONFIG, **config)
        self.framework = "torch"
        Policy.__init__(self, obs_space, action_space, config)
        LearningRateSchedule.__init__(self, config["lr"], config.get("lr_schedule"))
        self.n_agents = len(obs_space.original_space.spaces)
        config["model"]["n_agents"] = self.n_agents
        self.n_actions = action_space.spaces[0].n
        self.h_size = config["model"]["lstm_cell_size"]
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        self.reward_standardize = config["reward_standardize"]

        agent_obs_space = obs_space.original_space.spaces[0]
        if isinstance(agent_obs_space, Gym_Dict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError(
                    "Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError(
                        "Action mask shape must be {}, got {}".format(
                            (self.n_actions,), mask_shape))
                self.has_action_mask = True
            if "state" in space_keys:
                self.env_global_state_shape = _get_size(
                    agent_obs_space.spaces["state"])
                self.has_env_global_state = True
            else:
                self.env_global_state_shape = (self.obs_size, self.n_agents)
            # The real agent obs space is nested inside the dict
            config["model"]["full_obs_space"] = agent_obs_space
            agent_obs_space = agent_obs_space.spaces["obs"]
        else:
            self.obs_size = _get_size(agent_obs_space)
            self.env_global_state_shape = (self.obs_size, self.n_agents)

        core_arch = config["model"]["custom_model_config"]["model_arch_args"]["core_arch"]

        def create_model():
            return ModelCatalog.get_model_v2(
                agent_obs_space,
                action_space.spaces[0],
                self.n_actions,
                config["model"],
                framework="torch",
                name="model",
                default_model=JointQMLP if core_arch == "mlp" else JointQRNN)

        self.models = nn.ModuleDict(
            {f"agent_{i}": create_model() for i in range(self.n_agents)}
        ).to(self.device)
        self.aux_models = nn.ModuleDict(
            {f"agent_{i}": create_model() for i in range(self.n_agents)}
        ).to(self.device)
        self.aux_target_models = nn.ModuleDict(
            {f"agent_{i}": create_model() for i in range(self.n_agents)}
        ).to(self.device)

        self.exploration = self._create_exploration()

        self.cur_epsilon = 1.0
        self.update_target()  # initial sync

        # Setup optimizer
        self.params = list(self.models.parameters()) + list(self.aux_models.parameters())
        self.loss = JointQLoss(
            self.models, self.aux_models, self.aux_target_models, self.n_agents,
            self.n_actions, config["tdw_schedule"], self.device,
            self.config["double_q"], self.config["gamma"], config["lambda"]
        )

        if config["optimizer"] == "rmsprop":
            from torch.optim import RMSprop
            self.optimiser = RMSprop(
                params=self.params,
                lr=self.cur_lr)

        elif config["optimizer"] == "adam":
            from torch.optim import Adam
            self.optimiser = Adam(
                params=self.params,
                lr=self.cur_lr)

        else:
            raise ValueError("choose one optimizer type from rmsprop(RMSprop) or adam(Adam)")

    @property
    def _optimizers(self):
        return [self.optimiser]

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        explore = explore if explore is not None else self.config["explore"]
        obs_batch, action_mask, _, _ = self._unpack_observation(obs_batch)
        # We need to ensure we do not use the env global state
        # to compute actions

        # Compute actions for each agent
        with torch.no_grad():
            ma_q_values, ma_hiddens = [], []
            for i in range(self.n_agents):
                q_values, hiddens = _mac(
                    self.models[f"agent_{i}"],
                    torch.as_tensor(
                        obs_batch[:, i:i + 1], dtype=torch.float, device=self.device), [
                        torch.as_tensor(np.array(s)[:, i:i + 1], dtype=torch.float, device=self.device)
                        for s in state_batches
                    ])
                ma_q_values.append(q_values)
                ma_hiddens.append(hiddens)

            # aggregate individual agent values
            q_values = torch.cat(ma_q_values, dim=1)
            hiddens = []
            for i in range(len(state_batches)):
                ag_h = [ag_state[i] for ag_state in ma_hiddens]
                hiddens.append(
                    torch.cat(ag_h, dim=1)
                )

            avail = torch.as_tensor(action_mask, dtype=torch.float, device=self.device)
            masked_q_values = q_values.clone()
            masked_q_values[avail == 0.0] = -float("inf")
            masked_q_values_folded = torch.reshape(
                masked_q_values, [-1] + list(masked_q_values.shape)[2:])
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q_values_folded),
                timestep=timestep,
                explore=explore)
            actions = torch.reshape(
                actions,
                list(masked_q_values.shape)[:-1]).cpu().numpy()
            hiddens = [s.cpu().numpy() for s in hiddens]

            self.global_timestep += 1

        return tuple(actions.transpose([1, 0])), hiddens, {}

    @override(Policy)
    def compute_log_likelihoods(self,
                                actions,
                                obs_batch,
                                state_batches=None,
                                prev_action_batch=None,
                                prev_reward_batch=None):
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        return np.zeros(obs_batch.size()[0])

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state, terminal_flags = self._unpack_observation(
            samples[SampleBatch.CUR_OBS])
        (next_obs_batch, next_action_mask, next_env_global_state, _) = self._unpack_observation(
            samples[SampleBatch.NEXT_OBS])
        group_rewards = self._get_group_rewards(samples[SampleBatch.INFOS])

        input_list = [
            group_rewards, action_mask, next_action_mask,
            samples[SampleBatch.ACTIONS], samples[SampleBatch.DONES],
            obs_batch, next_obs_batch, terminal_flags
        ]
        if self.has_env_global_state:
            input_list.extend([env_global_state, next_env_global_state])

        output_list, _, seq_lens = \
            chop_into_sequences(
                episode_ids=samples[SampleBatch.EPS_ID],
                unroll_ids=samples[SampleBatch.UNROLL_ID],
                agent_indices=samples[SampleBatch.AGENT_INDEX],
                feature_columns=input_list,
                state_columns=[],  # RNN states not used here
                max_seq_len=self.config["model"]["max_seq_len"],
                dynamic_max=True)
        # These will be padded to shape [B * T, ...]
        if self.has_env_global_state:
            (rew, action_mask, next_action_mask, act, dones, obs, next_obs,
             env_global_state, next_env_global_state, terminal_flags) = output_list
        else:
            (rew, action_mask, next_action_mask, act, dones, obs,
             next_obs, terminal_flags) = output_list
        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            new_shape = [B, T] + list(arr.shape[1:])
            return torch.as_tensor(
                np.reshape(arr, new_shape), dtype=dtype, device=self.device)

        # reduce the scale of reward for small variance. This is also
        # because we copy the global reward to each agent in rllib_env
        rewards = to_batches(rew, torch.float) / self.n_agents
        if self.reward_standardize:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        actions = to_batches(act, torch.long)
        obs = to_batches(obs, torch.float).reshape(
            [B, T, self.n_agents, self.obs_size])
        action_mask = to_batches(action_mask, torch.float)
        next_obs = to_batches(next_obs, torch.float).reshape(
            [B, T, self.n_agents, self.obs_size])
        next_action_mask = to_batches(next_action_mask, torch.float)
        if self.has_env_global_state:
            env_global_state = to_batches(env_global_state, torch.float)
            next_env_global_state = to_batches(next_env_global_state,
                                               torch.float)

        terminated = to_batches(terminal_flags, torch.float)

        # Create mask for where index is < unpadded sequence length
        filled = np.reshape(
            np.tile(np.arange(T, dtype=np.float32), B),
            [B, T]) < np.expand_dims(seq_lens, 1)
        mask = torch.as_tensor(
            filled, dtype=torch.float, device=self.device).unsqueeze(2).expand(
            B, T, self.n_agents)

        # Compute loss
        loss_out, mask, masked_td_error, chosen_action_qvals, targets = (
            self.loss(self.global_timestep, rewards, actions, terminated, mask, obs, next_obs,
                      action_mask, next_action_mask, env_global_state,
                      next_env_global_state))

        # Optimise
        self.optimiser.zero_grad()
        loss_out.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.params, self.config["grad_norm_clipping"])
        self.optimiser.step()

        mask_elems = mask.sum().item()
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm
            if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() /
                            mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        return {LEARNER_STATS_KEY: stats}

    @override(Policy)
    def get_initial_state(self):  # initial RNN state
        return [
            s.expand([self.n_agents, -1]).cpu().numpy()
            for s in self.models["agent_0"].get_initial_state()
        ]

    @override(Policy)
    def get_weights(self):
        return {
            "models": self._cpu_dict(self.models.state_dict()),
            "aux_models": self._cpu_dict(self.aux_models.state_dict()),
            "aux_target_models": self._cpu_dict(self.aux_target_models.state_dict()),
        }

    @override(Policy)
    def set_weights(self, weights):
        self.models.load_state_dict(self._device_dict(weights["models"]))
        self.aux_models.load_state_dict(self._device_dict(weights["aux_models"]))
        self.aux_target_models.load_state_dict(self._device_dict(weights["aux_target_models"]))

    @override(Policy)
    def get_state(self):
        state = self.get_weights()
        state["cur_epsilon"] = self.cur_epsilon
        return state

    @override(Policy)
    def set_state(self, state):
        self.set_weights(state)
        self.set_epsilon(state["cur_epsilon"])

    def update_target(self):
        # self.aux_target_models.load_state_dict(self.aux_models.state_dict())
        soft_update(self.aux_target_models, self.aux_models, self.config["tau"])
        logger.debug("Updated target networks")

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon

    def _get_group_rewards(self, info_batch):
        group_rewards = np.array([
            info.get("_group_rewards", [0.0] * self.n_agents)
            for info in info_batch
        ])
        return group_rewards

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device)
            for k, v in state_dict.items()
        }

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def _unpack_observation(self, obs_batch):
        """Unpacks the observation, action mask, and state (if present)
        from agent grouping.

        Returns:
            obs (np.ndarray): obs tensor of shape [B, n_agents, obs_size]
            mask (np.ndarray): action mask, if any
            state (np.ndarray or None): state tensor of shape [B, state_size]
                or None if it is not in the batch
        """

        unpacked = _unpack_obs(
            np.array(obs_batch, dtype=np.float32),
            self.observation_space.original_space,
            tensorlib=np)

        unpacked_terminal_flag = None
        if isinstance(unpacked[0], dict):
            assert "obs" in unpacked[0] and "terminal" in unpacked[0]
            unpacked_obs = [
                np.concatenate(tree.flatten(u["obs"]), 1) for u in unpacked
            ]
            unpacked_terminal_flag = [
                np.concatenate(tree.flatten(u["terminal"]), 1) for u in unpacked
            ]
        else:
            unpacked_obs = unpacked

        obs = np.concatenate(
            unpacked_obs, axis=1
        ).reshape([len(obs_batch), self.n_agents, self.obs_size])
        if unpacked_terminal_flag:
            terminal_flag = np.concatenate(
                unpacked_terminal_flag, axis=1
            ).reshape([len(obs_batch), self.n_agents])
        else:
            terminal_flag = None

        if self.has_action_mask:
            action_mask = np.concatenate(
                [o["action_mask"] for o in unpacked], axis=1).reshape(
                [len(obs_batch), self.n_agents, self.n_actions])
        else:
            action_mask = np.ones(
                [len(obs_batch), self.n_agents, self.n_actions],
                dtype=np.float32)

        if self.has_env_global_state:
            state = np.concatenate(tree.flatten(unpacked[0]["state"]), 1)
        else:
            state = None
        return obs, action_mask, state, terminal_flag
