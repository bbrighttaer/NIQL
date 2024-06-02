from queue import Queue
from typing import Dict, Optional, TYPE_CHECKING

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.observation_function import ObservationFunction
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.utils.typing import TensorType

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

from niql.comm import InterAgentComm


class NIQLCallbacks(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_comm = InterAgentComm()
        self.battle_win_queue = Queue(maxsize=100)
        self.ally_survive_queue = Queue(maxsize=100)
        self.enemy_killing_queue = Queue(maxsize=100)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Runs when an episode is done.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (Dict[PolicyID, Policy]): Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (EnvID): Obsoleted: The ID of the environment, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """

        if self.legacy_callbacks.get("on_episode_end"):
            self.legacy_callbacks["on_episode_end"]({
                "env": base_env,
                "policy": policies,
                "episode": episode,
            })

        # Get current env from worker
        env = worker.env

        # SMAC metrics (from https://github.com/Replicable-MARL/MARLlib/blob/mq_dev/SMAC/metric/smac_callback.py)
        if hasattr(env, "env"):  # Needed because Marllib wraps the SMAC env
            env = env.env
            if hasattr(env, "death_tracker_ally") and hasattr(env, "death_tracker_enemy"):
                ally_state = env.death_tracker_ally
                enemy_state = env.death_tracker_enemy

                # count battle win rate in recent 100 games
                if self.battle_win_queue.full():
                    self.battle_win_queue.get()  # pop FIFO

                battle_win_this_episode = int(all(enemy_state == 1))  # all enemy died / win
                self.battle_win_queue.put(battle_win_this_episode)

                episode.custom_metrics["battle_win_rate"] = sum(
                    self.battle_win_queue.queue) / self.battle_win_queue.qsize()

                # count ally survive in recent 100 games
                if self.ally_survive_queue.full():
                    self.ally_survive_queue.get()  # pop FIFO

                ally_survive_this_episode = sum(ally_state == 0) / ally_state.shape[0]  # all enemy died / win
                self.ally_survive_queue.put(ally_survive_this_episode)

                episode.custom_metrics["ally_survive_rate"] = sum(
                    self.ally_survive_queue.queue) / self.ally_survive_queue.qsize()

                # count enemy killing rate in recent 100 games
                if self.enemy_killing_queue.full():
                    self.enemy_killing_queue.get()  # pop FIFO

                enemy_killing_this_episode = sum(enemy_state == 1) / enemy_state.shape[0]  # all enemy died / win
                self.enemy_killing_queue.put(enemy_killing_this_episode)

                episode.custom_metrics["enemy_kill_rate"] = sum(
                    self.enemy_killing_queue.queue) / self.enemy_killing_queue.qsize()


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
        # publish observation to other agents
        for agent_id, obs in agent_obs.items():
            policy_id = self.policy_mapping_fn(agent_id)
            policy = policies[policy_id]
            if hasattr(policy, "neighbour_messages") and policy.use_comm:
                all_n_obs = []
                for n_id, n_obs in agent_obs.items():
                    if n_id != agent_id:
                        n_policy_id = self.policy_mapping_fn(n_id)
                        n_policy = policies[n_policy_id]
                        message = n_policy.get_message(n_obs["obs"])
                        all_n_obs.append(message)
                policy.neighbour_messages = all_n_obs
        return agent_obs
