import logging
from functools import partial

import numpy as np
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils.torch_ops import huber_loss

from niql.algo.base_policy import NIQLBasePolicy
from niql.envs import DEBUG_ENVS
from niql.models import DQNModelsFactory
from niql.utils import to_numpy, tb_add_scalar, \
    tb_add_scalars, target_distribution_weighting, unroll_mac, unroll_mac_squeeze_wrapper, soft_update, tb_add_histogram

logger = logging.getLogger(__name__)


class IQLPolicyAttnComm(NIQLBasePolicy):

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config, DQNModelsFactory)

    def compute_trajectory_q_loss(self,
                                  rewards,
                                  is_weights,
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
                                  neighbour_next_obs=None,
                                  uniform_batch=None):
        """
        Computes the Q loss.
        Based on the JointQLoss of Marllib.

        Args:
            rewards: Tensor of shape [B, T]
            is_weights: Tensor of shape [B, T]
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
            uniform_batch: VAE training data
        """
        B, T = obs.shape[:2]

        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)
        whole_obs = whole_obs.unsqueeze(2)
        if neighbour_obs is not None:
            all_neighbour_msgs = torch.cat((neighbour_obs[:, 0:1], neighbour_next_obs), axis=1)
        else:
            all_neighbour_msgs = None
        tb_add_histogram(self, "batch_rewards", rewards)

        # Calculate estimated Q-Values
        mac_out, mac_out_h = unroll_mac_squeeze_wrapper(
            unroll_mac(
                self.model,
                whole_obs,
                self.comm_net,
                all_neighbour_msgs,
                partial(self.aggregate_messages, False)
            )
        )

        # Pick the Q-Values for the actions taken -> [B * n_agents, T]
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Calculate the Q-Values necessary for the target
        target_mac_out, target_mac_out_h = unroll_mac_squeeze_wrapper(
            unroll_mac(
                self.target_model,
                whole_obs,
                self.comm_net_target,
                all_neighbour_msgs,
                partial(self.aggregate_messages, True)
            )
        )
        target_mac_out = target_mac_out.detach()

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
        tb_add_histogram(self, "batch_targets", targets)

        # Get target distribution weights
        tdw_train_data = self.construct_tdw_dataset(uniform_batch)
        tdw_x = self.construct_tdw_dataset(SampleBatch({
            SampleBatch.OBS: obs.view(B * T, -1),
            SampleBatch.ACTIONS: actions.view(B * T, -1),
            SampleBatch.REWARDS: rewards.view(B * T, -1),
        }))
        tdw_weights = target_distribution_weighting(
            self, targets.detach().clone().view(B * T, -1),
        )
        tdw_weights = tdw_weights.view(B, T)
        # tdw_weights = self.get_tdw_weights(
        #     training_data=uniform_batch,
        #     obs=obs.view(B * T, -1),
        #     actions=actions.view(B * T, -1),
        #     rewards=rewards.view(B * T, -1),
        # )
        # tdw_weights = tdw_weights.view(*targets.shape)

        # Td-error
        td_delta = chosen_action_qvals - targets.detach()
        weights = is_weights * tdw_weights
        # weights /= torch.clamp(weights.max(), 1e-7)
        # weights = weights ** self.adaptive_gamma()
        td_error = td_delta * weights

        mask = mask.expand_as(td_error)
        self.model.tower_stats["td_error"] = to_numpy(td_delta * mask)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = huber_loss(masked_td_error).sum() / mask.sum()
        self.model.tower_stats["loss"] = to_numpy(loss)
        tb_add_scalar(self, "loss", loss.item())

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

        if self.use_comm and "comm_net" in weights:
            self.comm_net.load_state_dict(self._device_dict(weights["comm_net"]))
            self.comm_net_target.load_state_dict(self._device_dict(weights["comm_net_target"]))

            self.comm_aggregator.load_state_dict(self._device_dict(weights["comm_aggregator"]))
            self.comm_aggregator_target.load_state_dict(self._device_dict(weights["comm_aggregator_target"]))

    def get_weights(self):
        wts = {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "vae_model": self._cpu_dict(self.vae_model.state_dict()),
        }
        if self.use_comm:
            wts["comm_net"] = self._cpu_dict(self.comm_net.state_dict())
            wts["comm_net_target"] = self._cpu_dict(self.comm_net_target.state_dict())

            wts["comm_aggregator"] = self._cpu_dict(self.comm_aggregator.state_dict())
            wts["comm_aggregator_target"] = self._cpu_dict(self.comm_aggregator_target.state_dict())
        return wts

    def update_target(self):
        # self.target_model.load_state_dict(self.model.state_dict())
        # self.comm_net_target.load_state_dict(self.comm_net.state_dict())
        # self.comm_aggregator_target.load_state_dict(self.comm_aggregator.state_dict())
        self.target_model = soft_update(self.target_model, self.model, self.tau)
        if self.use_comm:
            self.comm_net_target = soft_update(self.comm_net_target, self.comm_net, self.tau)
            self.comm_aggregator_target = soft_update(self.comm_aggregator_target, self.comm_aggregator, self.tau)
        logger.debug("Updated target networks")

    def switch_models_to_eval_mode(self):
        # Switch to eval mode.
        if self.model:
            self.model.eval()

        if self.use_comm:
            self.comm_net.eval()
            self.comm_aggregator.eval()

    def switch_models_to_train_mode(self):
        # Set Model to train mode.
        if self.model:
            self.model.train()

        if self.use_comm:
            self.comm_net.train()
            self.comm_aggregator.train()
