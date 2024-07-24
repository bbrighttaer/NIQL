import logging
import random
import time
from functools import partial

import numpy as np
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils.torch_ops import huber_loss

from niql.algo.base_policy import NIQLBasePolicy
from niql.models import DQNModelsFactory
from niql.utils import to_numpy, tb_add_scalar, \
    target_distribution_weighting, unroll_mac, unroll_mac_squeeze_wrapper, soft_update, tb_add_histogram

logger = logging.getLogger(__name__)


def construct_vae_data(obs, prev_actions=None, n_actions=None):
    if prev_actions is not None and n_actions:
        actions_enc = torch.eye(n_actions).float().to(obs.device)[prev_actions.view(-1, )]
        obs = torch.cat([obs, actions_enc], dim=-1)
    return obs


class WIQL(NIQLBasePolicy):

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config, DQNModelsFactory)

    def compute_trajectory_q_loss(self,
                                  rewards,
                                  weights,
                                  actions,
                                  prev_actions,
                                  terminated,
                                  mask,
                                  obs,
                                  next_obs,
                                  action_mask,
                                  next_action_mask,
                                  state=None,
                                  next_state=None,
                                  neighbour_msg=None,
                                  neighbour_next_msg=None,
                                  shared_msg=None,
                                  uniform_batch=None):
        """
        Computes the Q loss.
        Based on the JointQLoss of Marllib.

        Args:
            rewards: Tensor of shape [B, T]
            weights: Tensor of shape [B, T]
            actions: Tensor of shape [B, T]
            prev_actions: Tensor of shape [B, T]
            terminated: Tensor of shape [B, T]
            mask: Tensor of shape [B, T]
            obs: Tensor of shape [B, T, obs_size]
            next_obs: Tensor of shape [B, T, obs_size]
            action_mask: Tensor of shape [B, T, n_actions]
            next_action_mask: Tensor of shape [B, T, n_actions]
            state: Tensor of shape [B, T, state_dim] (optional)
            next_state: Tensor of shape [B, T, state_dim] (optional)
            neighbour_msg: Tensor of shape [B, T, num_neighbours, obs_size]
            neighbour_next_msg: Tensor of shape [B, T, num_neighbours, obs_size]
            shared_msg: Tensor of shape [B, T, comm_size]
            uniform_batch: VAE training data
        """
        B, T = obs.shape[:2]

        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)
        whole_obs = whole_obs.unsqueeze(2)

        if neighbour_msg is not None:
            all_neighbour_msgs = torch.cat((neighbour_msg[:, 0:1], neighbour_next_msg), axis=1)
        else:
            all_neighbour_msgs = None
        tb_add_histogram(self, "batch_rewards", rewards)

        kwargs = {}
        if self.add_action_to_obs:
            kwargs["prev_actions"] = torch.cat([prev_actions, actions[:, -1:]], dim=-1)
            kwargs["n_actions"] = self.n_actions

        # VAE data
        # vae_data = construct_vae_data(obs.view(B * T, -1), prev_actions, kwargs.get("n_actions"))

        # Calculate estimated Q-Values
        mac_out, mac_out_h = unroll_mac_squeeze_wrapper(
            unroll_mac(
                self.model,
                whole_obs,
                all_neighbour_msgs,
                partial(self.aggregate_messages, False),
                **kwargs,
            )
        )
        msg_mac_out = mac_out[:, :, self.n_actions:]
        mac_out = mac_out[:, :, : self.n_actions]

        # Pick the Q-Values for the actions taken -> [B * n_agents, T]
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Pick selected messages' q-values
        chosen_msg_actions = shared_msg.max(dim=2)[1]
        chosen_msg_qvals = torch.gather(msg_mac_out[:, :-1], dim=2, index=chosen_msg_actions.unsqueeze(2)).squeeze(2)

        # Calculate the Q-Values necessary for the target
        target_mac_out, target_mac_out_h = unroll_mac_squeeze_wrapper(
            unroll_mac(
                self.target_model,
                whole_obs,
                all_neighbour_msgs,
                partial(self.aggregate_messages, True),
                **kwargs
            )
        )
        target_mac_out = target_mac_out.detach()
        msg_target_mac_out = target_mac_out[:, :, self.n_actions:]
        target_mac_out = target_mac_out[:, :, : self.n_actions]

        # we only need target_mac_out for raw next_obs part
        target_mac_out = target_mac_out[:, 1:]
        msg_target_mac_out = msg_target_mac_out[:, 1:]

        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        # Max over target Q-Values
        if self.config["double_q"]:
            mac_out_tp1 = mac_out.clone().detach()
            mac_out_tp1 = mac_out_tp1[:, 1:]
            msg_mac_out_tp1 = msg_mac_out[:, 1:]

            # mask out unallowed actions
            mac_out_tp1[ignore_action_tp1] = -np.inf

            # obtain best actions at t+1 according to policy NN
            cur_max_actions = mac_out_tp1.argmax(dim=2, keepdim=True)
            msg_cur_max_actions = msg_mac_out_tp1.argmax(dim=2, keepdim=True)

            # use the target network to estimate the Q-values of policy
            # network's selected actions
            target_max_qvals = torch.gather(target_mac_out, 2, cur_max_actions).squeeze(2)
            msg_target_max_qvals = torch.gather(msg_target_mac_out, 2, msg_cur_max_actions).squeeze(2)
        else:
            target_max_qvals = target_mac_out.max(dim=2)[0]
            msg_target_max_qvals = msg_target_mac_out.max(dim=2)[0]

        # Calculate Q-Learning targets
        targets = rewards + self.config["gamma"] * (1 - terminated) * target_max_qvals
        tb_add_histogram(self, "batch_targets", targets)

        # Calculate msg Q-learning targets
        msg_targets = rewards + self.config["gamma"] * (1 - terminated) * msg_target_max_qvals

        # Construct data for training tdw vae
        # vae_data = self.construct_tdw_dataset(SampleBatch({
        #     SampleBatch.OBS: obs.view(B * T, -1),
        #     SampleBatch.ACTIONS: actions.view(B * T, -1),
        #     SampleBatch.REWARDS: targets.view(B * T, -1),
        #     SampleBatch.NEXT_OBS: next_obs.view(B * T, -1),
        # }))

        # Get target distribution weights
        # if self.global_timestep < self.config["tdw_warm_steps"]:
        #     start = time.perf_counter()
        #     self.fit_vae(vae_data, num_epochs=2)
        #     tb_add_scalar(self, "fit_vae_time", time.perf_counter() - start)
        # elif random.random() < self.tdw_schedule.value(self.global_timestep):
        #     start = time.perf_counter()
        #     tdw_weights = self.get_tdw_weights(vae_data, targets)
        #     tdw_weights = tdw_weights.view(B, T)
        #     weights = tdw_weights ** 1.0  # self.adaptive_gamma()
        #     tb_add_scalar(self, "tdw_time", time.perf_counter() - start)
        # else:
        weights = torch.ones_like(targets)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()
        msg_td_error = chosen_msg_qvals - msg_targets.detach()

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        masked_msg_td_error = msg_td_error * mask
        self.model.tower_stats["td_error"] = to_numpy((masked_td_error + masked_msg_td_error) / 2.)

        # Calculate loss
        loss = weights * huber_loss(masked_td_error)
        loss = loss.sum() / mask.sum()
        msg_loss = huber_loss(masked_msg_td_error)
        msg_loss = msg_loss.sum() / mask.sum()
        self.model.tower_stats["loss"] = loss.item()
        tb_add_scalar(self, "loss", loss.item())
        tb_add_scalar(self, "msg_loss", msg_loss.item())
        loss += msg_loss

        # vae loss
        # vae_loss = self.compute_vae_loss(vae_data)
        # lamda = 0.7
        # loss = loss + lamda * vae_loss
        # tb_add_scalar(self, "joint_loss", loss.item())
        # tb_add_scalar(self, "vae_loss", vae_loss.item())

        # gather td error for each unique target for analysis (matrix game case - discrete reward)
        # if self.config.get("env_name") in DEBUG_ENVS:
        #     unique_targets = torch.unique(targets.int())
        #     mean_td_stats = {
        #         t.item(): torch.mean(torch.abs(masked_td_error).view(-1, )[targets.view(-1, ).int() == t]).item()
        #         for t in unique_targets
        #     }
        #     tb_add_scalars(self, "td-error_dist", mean_td_stats)

        return loss, mask, masked_td_error, chosen_action_qvals, targets

    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))
        self.vae_model.load_state_dict(self._device_dict(weights["vae_model"]))

        if self.use_comm:
            self.comm_aggregator.load_state_dict(self._device_dict(weights["comm_aggregator"]))
            self.comm_aggregator_target.load_state_dict(self._device_dict(weights["comm_aggregator_target"]))

    def get_weights(self):
        wts = {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "vae_model": self._cpu_dict(self.vae_model.state_dict()),
            "target_vae_model": self._cpu_dict(self.target_vae_model.state_dict()),
        }
        if self.use_comm:
            wts["comm_aggregator"] = self._cpu_dict(self.comm_aggregator.state_dict())
            wts["comm_aggregator_target"] = self._cpu_dict(self.comm_aggregator_target.state_dict())
        return wts

    def update_target(self):
        self.target_model = soft_update(self.target_model, self.model, self.tau)
        self.target_vae_model = soft_update(self.target_vae_model, self.vae_model, self.tau)
        if self.use_comm:
            self.comm_aggregator_target = soft_update(self.comm_aggregator_target, self.comm_aggregator, self.tau)
        logger.debug("Updated target networks")

    def switch_models_to_eval_mode(self):
        # Switch to eval mode.
        if self.model:
            self.model.eval()

        if self.use_comm:
            self.comm_aggregator.eval()

    def switch_models_to_train_mode(self):
        # Set Model to train mode.
        if self.model:
            self.model.train()

        if self.use_comm:
            self.comm_aggregator.train()
