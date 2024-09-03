from queue import Queue
from typing import Dict, Optional, TYPE_CHECKING

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv, GroupAgentsWrapper
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from niql.envs import RLlibSMAC

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker


class NIQLCallbacks(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        # clear shared local messages
        for policy in list(policies.values()):
            if hasattr(policy, "initialise_messages_hist"):
                policy.initialise_messages_hist()

        # Get current env from worker
        env = worker.env
        if isinstance(env, GroupAgentsWrapper):
            env = env.env

        # SMAC metrics (from https://github.com/Replicable-MARL/MARLlib/blob/mq_dev/SMAC/metric/smac_callback.py)
        if isinstance(env, RLlibSMAC):
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
