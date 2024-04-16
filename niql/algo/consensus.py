from typing import List, Any

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import PolicyID


class ConsensusUpdate:
    """
    Called after the training step to trigger consensus among agents.
    """

    def __init__(self,
                 workers: WorkerSet,
                 consensus_update_freq: int,
                 policies: List[PolicyID] = frozenset([])):
        self.workers = workers
        self.local_worker = workers.local_worker()
        self.consensus_update_freq = consensus_update_freq
        self.policies = policies

    def __call__(self, _: Any) -> None:
        to_update = self.policies or self.local_worker.policies_to_train
        self.workers.local_worker().foreach_trainable_policy(
            lambda p, p_id: p_id in to_update and hasattr(p, 'start_consensus') and p.start_consensus(p_id, {
                k: v for k, v in self.workers.local_worker().policy_map.items() if k != p_id
            })
        )
