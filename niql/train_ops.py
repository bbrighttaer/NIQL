"""
From RLLib.

Modifications:
1. Changed how STEPS_TRAINED_COUNTER is counted; increment by 1 instead of by number of samples used for training
"""

import logging
from typing import List, Any

import ray
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import \
    AGENT_STEPS_TRAINED_COUNTER, LAST_TARGET_UPDATE_TS, LEARN_ON_BATCH_TIMER, \
    NUM_TARGET_UPDATES, STEPS_SAMPLED_COUNTER, \
    STEPS_TRAINED_COUNTER, WORKER_UPDATE_TIMER, _check_sample_batch_type, \
    _get_global_vars, _get_shared_metrics
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.typing import PolicyID, SampleBatchType

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


class TrainOneStep:
    """Callable that improves the policy and updates workers.

    This should be used with the .for_each() operator. A tuple of the input
    and learner stats will be returned.

    Examples:
        >>> rollouts = ParallelRollouts(...)
        >>> train_op = rollouts.for_each(TrainOneStep(workers))
        >>> print(next(train_op))  # This trains the policy on one batch.
        SampleBatch(...), {"learner_stats": ...}

    Updates the STEPS_TRAINED_COUNTER counter and LEARNER_INFO field in the
    local iterator context.
    """

    def __init__(self,
                 workers: WorkerSet,
                 policies: List[PolicyID] = frozenset([]),
                 num_sgd_iter: int = 1,
                 sgd_minibatch_size: int = 0):
        self.workers = workers
        self.local_worker = workers.local_worker()
        self.policies = policies
        self.num_sgd_iter = num_sgd_iter
        self.sgd_minibatch_size = sgd_minibatch_size

    def __call__(self,
                 batch: SampleBatchType) -> (SampleBatchType, List[dict]):
        _check_sample_batch_type(batch)
        metrics = _get_shared_metrics()
        learn_timer = metrics.timers[LEARN_ON_BATCH_TIMER]
        with learn_timer:
            # Subsample minibatches (size=`sgd_minibatch_size`) from the
            # train batch and loop through train batch `num_sgd_iter` times.
            if self.num_sgd_iter > 1 or self.sgd_minibatch_size > 0:
                lw = self.workers.local_worker()
                learner_info = do_minibatch_sgd(
                    batch, {
                        pid: lw.get_policy(pid)
                        for pid in self.policies
                                   or self.local_worker.policies_to_train
                    }, lw, self.num_sgd_iter, self.sgd_minibatch_size, [])
            # Single update step using train batch.
            else:
                learner_info = \
                    self.workers.local_worker().learn_on_batch(batch)

            metrics.info[LEARNER_INFO] = learner_info
            learn_timer.push_units_processed(batch.count)
        metrics.counters[STEPS_TRAINED_COUNTER] += 1  # batch.count
        if isinstance(batch, MultiAgentBatch):
            metrics.counters[
                AGENT_STEPS_TRAINED_COUNTER] += batch.agent_steps()
        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with metrics.timers[WORKER_UPDATE_TIMER]:
                weights = ray.put(self.workers.local_worker().get_weights(
                    self.policies or self.local_worker.policies_to_train))
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights, _get_global_vars())
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())
        return batch, learner_info


class UpdateTargetNetwork:
    """Periodically call policy.update_target() on all trainable policies.

    This should be used with the .for_each() operator after training step
    has been taken.

    Examples:
        >>> train_op = ParallelRollouts(...).for_each(TrainOneStep(...))
        >>> update_op = train_op.for_each(
        ...     UpdateTargetIfNeeded(workers, target_update_freq=500))
        >>> print(next(update_op))
        None

    Updates the LAST_TARGET_UPDATE_TS and NUM_TARGET_UPDATES counters in the
    local iterator context. The value of the last update counter is used to
    track when we should update the target next.
    """

    def __init__(self,
                 workers: WorkerSet,
                 target_update_freq: int,
                 by_steps_trained: bool = False,
                 policies: List[PolicyID] = frozenset([])):
        self.workers = workers
        self.local_worker = workers.local_worker()
        self.target_update_freq = target_update_freq
        self.policies = policies
        if by_steps_trained:
            self.metric = STEPS_TRAINED_COUNTER
        else:
            self.metric = STEPS_SAMPLED_COUNTER

    def __call__(self, _: Any) -> None:
        metrics = _get_shared_metrics()
        cur_ts = metrics.counters[self.metric]
        last_update = metrics.counters[LAST_TARGET_UPDATE_TS]
        if cur_ts - last_update >= self.target_update_freq:
            to_update = self.policies or self.local_worker.policies_to_train
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, p_id: p_id in to_update and p.update_target())
            metrics.counters[NUM_TARGET_UPDATES] += 1
            metrics.counters[LAST_TARGET_UPDATE_TS] = cur_ts
