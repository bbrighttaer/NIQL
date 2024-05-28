import torch
from marllib.marl.algos.core.VD.iql_vdn_qmix import JointQPolicy as Policy
from ray.rllib.agents.qmix.qmix_policy import _mac
from ray.rllib.execution.replay_buffer import *
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils import override


class JointQPolicy(LearningRateSchedule, Policy):

    def __init__(self, obs_space, action_space, config):
        Policy.__init__(self, obs_space, action_space, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        self.joint_q_values = []

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
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        # We need to ensure we do not use the env global state
        # to compute actions

        # Compute actions
        with torch.no_grad():
            q_values, hiddens = _mac(
                self.model,
                torch.as_tensor(
                    obs_batch, dtype=torch.float, device=self.device), [
                    torch.as_tensor(
                        np.array(s), dtype=torch.float, device=self.device)
                    for s in state_batches
                ])
            avail = torch.as_tensor(
                action_mask, dtype=torch.float, device=self.device)
            masked_q_values = q_values.clone()
            masked_q_values[avail == 0.0] = -float("inf")
            masked_q_values_folded = torch.reshape(masked_q_values, [-1] + list(masked_q_values.shape)[2:])
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q_values_folded),
                timestep=timestep,
                explore=explore,
            )
            actions = torch.reshape(
                actions,
                list(masked_q_values.shape)[:-1]).cpu().numpy()
            hiddens = [s.cpu().numpy() for s in hiddens]
            # self.joint_q_values = masked_q_values_folded.numpy().tolist()

        return tuple(actions.transpose([1, 0])), hiddens, {}
