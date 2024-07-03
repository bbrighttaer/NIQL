# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from marllib.marl.algos.utils.episode_replay_buffer import EpisodeBasedReplayBuffer
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
from torch.utils.tensorboard import SummaryWriter

from niql.replay_buffers import EpisodeBasedReplayBuffer
from niql.utils import get_priority_update_func


def episode_execution_plan(trainer: Trainer, workers: WorkerSet,
                           config: TrainerConfigDict, **kwargs) -> LocalIterator[dict]:
    """Execution plan of the DQN algorithm. Defines the distributed dataflow.

    Args:
        trainer (Trainer): The Trainer object creating the execution plan.
        workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
            of the Trainer.
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        LocalIterator[dict]: A local iterator over training metrics.
    """
    summary_writer = SummaryWriter()
    local_replay_buffer = EpisodeBasedReplayBuffer(
        learning_starts=config["learning_starts"],
        capacity=config["buffer_size"],
        replay_batch_size=config["train_batch_size"],
        replay_sequence_length=config.get("replay_sequence_length", 1),
        replay_burn_in=config.get("burn_in", 0),
        replay_zero_init_states=config.get("zero_init_states", True),
        enable_stochastic_eviction=config.get("enable_stochastic_eviction", False)
    )

    # Assign to Trainer, so we can store the LocalReplayBuffer's
    # data when we save checkpoints.
    trainer.local_replay_buffer = local_replay_buffer

    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our local replay buffer. Calling
    # next() on store_op drives this.
    store_op = rollouts.for_each(
        StoreToReplayBuffer(local_buffer=local_replay_buffer))

    # (2) Read and train on experiences from the replay buffer. Every batch
    # returned from the LocalReplay() iterator is passed to TrainOneStep to
    # take a SGD step, and then we decide whether to update the target network.
    post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)

    train_step_op = TrainOneStep(workers)
    policy_map = workers.local_worker().policy_map

    # replay_op = Replay(local_buffer=local_replay_buffer) \
    #     .for_each(lambda x: post_fn(x, workers, config,
    #                                 policy_map=policy_map,
    #                                 summary_writer=summary_writer,
    #                                 replay_buffer=local_replay_buffer,
    #                                 )) \
    #     .for_each(train_step_op) \
    #     .for_each(get_priority_update_func(local_replay_buffer, config)) \
    #     .for_each(UpdateTargetNetwork(workers, config.get("target_network_update_freq", 200)))

    replay_op = Replay(local_buffer=local_replay_buffer) \
        .for_each(lambda x: post_fn(x, workers, config,
                                    policy_map=policy_map,
                                    summary_writer=summary_writer,
                                    replay_buffer=local_replay_buffer,
                                    )) \
        .for_each(train_step_op) \
        .for_each(UpdateTargetNetwork(workers, config.get("target_network_update_freq", 200)))

    # Alternate deterministically between (1) and (2). Only return the output
    # of (2) since training metrics are not available until (2) runs.
    train_op = Concurrently(
        [store_op, replay_op],
        mode="round_robin",
        output_indexes=[1],
        round_robin_weights=[1, 1])

    return StandardMetricsReporting(train_op, workers, config)
