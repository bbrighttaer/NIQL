from typing import Dict, Optional

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import PolicyID


class NIQLCallbacks(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:
        episode.user_data["q_values"] = {}
        for policy_id in policies:
            if policy_id != DEFAULT_POLICY_ID:
                episode.user_data["q_values"][f"{policy_id}/q_values"] = []

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:
        for i, (policy_id, policy) in enumerate(policies.items()):
            if hasattr(policy, "q_values"):
                episode.user_data["q_values"][f"{policy_id}/q_values"].append(policy.q_values)

            # single policy case
            elif hasattr(policy, "joint_q_values"):
                for j, q_values in enumerate(policy.joint_q_values):
                    key = f"policy_{j}/q_values"
                    if key not in episode.user_data["q_values"]:
                        episode.user_data["q_values"][key] = []
                    episode.user_data["q_values"][key].append(q_values)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        for key, value in episode.user_data["q_values"].items():
            episode.hist_data[key] = episode.user_data["q_values"][key]
