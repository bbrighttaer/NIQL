import logging
from functools import partial

import numpy as np
import torch
from ray.rllib import Policy
from ray.rllib.utils import override
from ray.rllib.utils.torch_ops import huber_loss

from niql.algo.base_policy import NIQLBasePolicy
from niql.envs import DEBUG_ENVS
from niql.models import BQLModelsFactory
from niql.utils import unroll_mac, \
    unroll_mac_squeeze_wrapper, to_numpy, soft_update, tb_add_scalar, tb_add_scalars, target_distribution_weighting

logger = logging.getLogger(__name__)


class WIBQLPolicy(NIQLBasePolicy):
    """
    Implementation of Weighted Independent Best Possible Q-learning objective
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config, BQLModelsFactory)

    def compute_trajectory_q_loss(self,
                                  rewards,
                                  is_weights,
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
                                  neighbour_obs=None,
                                  neighbour_next_obs=None,
                                  uniform_batch=None):
        """
        Computes the Q loss.

        Args:
            rewards: Tensor of shape [B, T]
            is_weights: Tensor of shape [B, T]
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

        kwargs = {}
        if self.add_action_to_obs:
            kwargs["prev_actions"] = torch.cat([prev_actions, actions[:, -1:]], dim=-1)
            kwargs["n_actions"] = self.n_actions

        # Auxiliary encoder objective
        # Qe(s, a_i)
        qe_out, qe_h_out = unroll_mac_squeeze_wrapper(
            unroll_mac(
                self.auxiliary_model,
                whole_obs,
                self.comm_net,
                all_neighbour_msgs,
                partial(self.aggregate_messages, False),
                **kwargs,
            )
        )
        qe_qvals = torch.gather(qe_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Qi(s', a'_i*)
        qi_out, qi_h_out = unroll_mac_squeeze_wrapper(
            unroll_mac(
                self.model,
                whole_obs,
                self.comm_net,
                all_neighbour_msgs,
                partial(self.aggregate_messages, False),
                **kwargs,
            )
        )
        # qi_out is used for Qi(s,a) objective, we clone it here to avoid setting values to -np.inf
        qi_out_sp = qi_out[:, 1:].clone()
        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        qi_out_sp[ignore_action_tp1] = -np.inf
        qi_out_sp_qvals = qi_out_sp.max(dim=2)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.config["gamma"] * (1 - terminated) * qi_out_sp_qvals

        # Get target distribution weights
        # tdw_weights = target_distribution_weighting(
        #     self, targets.detach().clone().view(B * T, -1),
        # )
        # tdw_weights = tdw_weights.view(B, T)
        tdw_weights = self.get_tdw_weights(targets.detach().clone())
        tdw_weights = tdw_weights.view(*targets.shape)

        # Qe_i TD error
        td_error = qe_qvals - targets.detach()

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        self.model.tower_stats["td_error"] = to_numpy(masked_td_error)

        # Qe loss
        qe_loss = tdw_weights * is_weights * huber_loss(masked_td_error)
        qe_loss = qe_loss.sum() / mask.sum()
        self.model.tower_stats["Qe_loss"] = to_numpy(qe_loss)
        tb_add_scalar(self, "qe_loss", qe_loss.item())

        # gather td error for each unique target for analysis (matrix game case - discrete reward)
        if self.config.get("env_name") in DEBUG_ENVS:
            unique_targets = torch.unique(targets.int())
            mean_td_stats = {
                t.item(): torch.mean(torch.abs(masked_td_error).view(-1, )[targets.view(-1, ).int() == t]).item()
                for t in unique_targets
            }
            tb_add_scalars(self, "td-error_dist", mean_td_stats)

        # Output of Qe_bar
        qe_bar_out, qe_bar_h_out = unroll_mac_squeeze_wrapper(
            unroll_mac(
                self.auxiliary_model_target,
                whole_obs,
                self.comm_net_target,
                all_neighbour_msgs,
                partial(self.aggregate_messages, True),
                **kwargs,
            )
        )
        qe_bar_out = qe_bar_out.detach()

        # Qi function objective
        # Qi(s, a)
        qi_out_s_qvals = torch.gather(qi_out[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        # Qe_bar(s, a)
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

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.auxiliary_model.load_state_dict(self._device_dict(weights["auxiliary_model"]))
        self.auxiliary_model_target.load_state_dict(self._device_dict(weights["auxiliary_model_target"]))

        if self.use_comm and "comm_net" in weights:
            self.comm_net.load_state_dict(self._device_dict(weights["comm_net"]))
            self.comm_net_target.load_state_dict(self._device_dict(weights["comm_net_target"]))

            self.comm_aggregator.load_state_dict(self._device_dict(weights["comm_aggregator"]))
            self.comm_aggregator_target.load_state_dict(self._device_dict(weights["comm_aggregator_target"]))

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

            wts["comm_aggregator"] = self._cpu_dict(self.comm_aggregator.state_dict())
            wts["comm_aggregator_target"] = self._cpu_dict(self.comm_aggregator_target.state_dict())
        return wts

    def update_target(self):
        self.auxiliary_model_target = soft_update(self.auxiliary_model_target, self.auxiliary_model, self.tau)
        if self.use_comm:
            self.comm_net_target = soft_update(self.comm_net_target, self.comm_net, self.tau)
            self.comm_aggregator_target = soft_update(self.comm_aggregator_target, self.comm_aggregator, self.tau)

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
            self.auxiliary_model.train()

        if self.use_comm:
            self.comm_net.train()
            self.comm_aggregator.train()
