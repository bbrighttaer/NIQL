import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from marllib import marl
from niql import envs, config

os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
seed = 321
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-a', '--algo',
        type=str,
        default='qmix',
        choices=['vdn', 'qmix'],
        help='Select which CTDE algorithm to run.',
    )
    args = parser.parse_args()

    # get env
    env = envs.make_matrix_game_env()

    # initialise algorithm with hyperparameters
    if args.algo == 'qmix':
        algo = marl.algos.qmix(hyperparam_source="mpe")
    else:
        algo = marl.algos.vdn(hyperparam_source="mpe")
    algo.algo_parameters = config.COOP_MATRIX['algo_parameters']

    # build model
    model = marl.build_model(env, algo, model_preference=config.COOP_MATRIX['model_preference'])

    # start training
    gpu_count = torch.cuda.device_count()
    algo.fit(
        env,
        model,
        stop=config.COOP_MATRIX['stop_condition'],
        local_mode=gpu_count == 0,
        num_gpus=gpu_count,
        num_workers=0,
        share_policy='all',
        checkpoint_freq=10,
    )
