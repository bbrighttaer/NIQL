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
        default='vdn',
        choices=['vdn', 'qmix'],
        help='Select which CTDE algorithm to run.',
    )
    args = parser.parse_args()

    # get env
    env = envs.make_mpe_env()

    # initialise algorithm with hyperparameters
    if args.algo == 'qmix':
        algo = marl.algos.qmix(hyperparam_source="mpe")
    else:
        algo = marl.algos.vdn(hyperparam_source="mpe")
    algo.algo_parameters = config.mpe['algo_parameters']

    # build model
    model = marl.build_model(env, algo, model_preference=config.mpe['model_preference'])

    # start training
    gpu_count = torch.cuda.device_count()
    algo.fit(
        env,
        model,
        stop=config.mpe['stop_condition'],
        local_mode=gpu_count == 0,
        num_gpus=gpu_count,
        num_workers=0,
        share_policy='all',
        checkpoint_freq=10,
    )

    # rendering
    # r_path = 'exp_results/qmix_mlp_simple_spread/QMIX_grouped_mpe_simple_spread_697a8_00000_0_2024-03-13_02-21-34/checkpoint_003850'
    # p_path = 'exp_results/qmix_mlp_simple_spread/QMIX_grouped_mpe_simple_spread_697a8_00000_0_2024-03-13_02-21-34/params.json'
    # algo.render(env, model,
    #             restore_path={'params_path': p_path,  # experiment configuration
    #                           'model_path': r_path,  # checkpoint path
    #                           'render': True},  # render
    #             local_mode=True,
    #             share_policy="all",
    #             checkpoint_end=False,
    #             )
