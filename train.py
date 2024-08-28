import logging
import os
from argparse import ArgumentParser

import torch
from marllib.marl import build_model

from niql import algos, scripts, envs
from niql.algo import ALGORITHMS

logger = logging.getLogger(__name__)

os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"


def register_algorithms():
    # register execution script
    for a in ALGORITHMS:
        algos.register_algo(
            algo_name=a,
            style="vd",
            script=scripts.run_joint_q,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a", "--algo",
        type=str,
        default="vdn",
        choices=ALGORITHMS,
        help="Select which CTDE algorithm to run.",
    )

    args = parser.parse_args()

    # get env
    env, exp_config = envs.get_active_env()

    register_algorithms()

    # initialise algorithm with hyperparameters
    algo = getattr(algos, args.algo)
    algo.algo_parameters = exp_config["algo_parameters"]

    # build model
    model_config = exp_config["model_preference"]
    model = build_model(env, algo, model_preference=exp_config["model_preference"])

    gpu_count = torch.cuda.device_count()

    # start training
    algo.fit(
        env,
        model,
        stop=exp_config["stop_condition"],
        local_mode=True,
        num_gpus=gpu_count,
        num_workers=1,
        share_policy="all",
        checkpoint_freq=10000,
    )
