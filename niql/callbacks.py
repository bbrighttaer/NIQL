from copy import deepcopy
from typing import Dict, Optional

from ray.rllib import SampleBatch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.evaluation.observation_function import ObservationFunction
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import PolicyID, AgentID, TensorType

from niql.comm import InterAgentComm, Message, CommMsg


class NIQLCallbacks(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_comm = InterAgentComm()

    # def on_episode_start(self,
    #                      *,
    #                      worker: "RolloutWorker",
    #                      base_env: BaseEnv,
    #                      policies: Dict[PolicyID, Policy],
    #                      episode: MultiAgentEpisode,
    #                      env_index: Optional[int] = None,
    #                      **kwargs) -> None:
    #     # comm setup
    #     for policy_id, policy in policies.items():
    #         if hasattr(policy, "comm") and policy.comm is None:
    #             policy.comm = self.agent_comm
    #             self.agent_comm.register(policy_id, policy)
    #
    #         if hasattr(policy, "policy_id") and policy.policy_id is None:
    #             policy.policy_id = policy_id
    #
    #     episode.user_data["q_values"] = {}
    #     for policy_id in policies:
    #         if policy_id != DEFAULT_POLICY_ID:
    #             episode.user_data["q_values"][f"{policy_id}/q_values"] = []
    #
    # def on_episode_step(self,
    #                     *,
    #                     worker: "RolloutWorker",
    #                     base_env: BaseEnv,
    #                     policies: Optional[Dict[PolicyID, Policy]] = None,
    #                     episode: MultiAgentEpisode,
    #                     env_index: Optional[int] = None,
    #                     **kwargs) -> None:
    #     for i, (policy_id, policy) in enumerate(policies.items()):
    #         if hasattr(policy, "q_values"):
    #             episode.user_data["q_values"][f"{policy_id}/q_values"].append(policy.q_values)
    #
    #         # single policy case
    #         elif hasattr(policy, "joint_q_values"):
    #             for j, q_values in enumerate(policy.joint_q_values):
    #                 key = f"policy_{j}/q_values"
    #                 if key not in episode.user_data["q_values"]:
    #                     episode.user_data["q_values"][key] = []
    #                 episode.user_data["q_values"][key].append(q_values)
    #
    # def on_episode_end(self,
    #                    *,
    #                    worker: "RolloutWorker",
    #                    base_env: BaseEnv,
    #                    policies: Dict[PolicyID, Policy],
    #                    episode: MultiAgentEpisode,
    #                    env_index: Optional[int] = None,
    #                    **kwargs) -> None:
    #     for key, value in episode.user_data["q_values"].items():
    #         episode.hist_data[key] = episode.user_data["q_values"][key]

    # def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
    #     super().on_train_result(trainer=trainer, result=result, **kwargs)
    #     config = trainer.config
    #
    #     if config.get("algorithm", "").lower() == "dbql":
    #         # retrieve replay buffers of all agents/policies
    #         policy_to_buffer = trainer.local_replay_buffer.replay_buffers
    #
    #         # retrieve policies
    #         policies = trainer.workers.local_worker().policy_map
    #
    #         # clear state-value buffers of all agents
    #         for buffer in policy_to_buffer.values():
    #             buffer.clear_state_value_buffer()
    #
    #         # share data among agents and store in their buffers
    #         for policy_id, buffer in policy_to_buffer.items():
    #             # sample local experiences
    #             batch = buffer.sample_local_experiences(config["sharing_batch_size"])
    #
    #             # compose state-value tuples
    #             batch = policies[policy_id].compute_state_values_from_batch(batch)
    #
    #             # send to other agents
    #             for agent, n_buffer in policy_to_buffer.items():
    #                 if agent != policy_id:
    #                     n_buffer.add_shared_state_value_batch(batch.copy())


class ObservationCommWrapper(ObservationFunction):
    """
    Facilitates inter agent communication in each time step.
    """

    def __init__(self, policy_mapping_fn):
        self.policy_mapping_fn = policy_mapping_fn

    def __call__(self, agent_obs: Dict[AgentID, TensorType],
                 worker: RolloutWorker, base_env: BaseEnv,
                 policies: Dict[PolicyID, Policy], episode: MultiAgentEpisode,
                 **kw) -> Dict[AgentID, TensorType]:
        if hasattr(worker.callbacks, 'agent_comm'):
            # get communication channel
            comm = worker.callbacks.agent_comm

            # publish observation to other agents
            for agent_id, obs in agent_obs.items():
                policy_id = self.policy_mapping_fn(agent_id)
                comm.broadcast(
                    policy_id,
                    Message(msg_type=CommMsg.OBSERVATION, msg=deepcopy(agent_obs[agent_id]))
                )
        return agent_obs
