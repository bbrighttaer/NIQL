import os
import random
from argparse import ArgumentParser

import torch
from marllib import marl
from niql import envs, scripts, config
import numpy as np

os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
seed = 321
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '-m', '--model_arch',
        default='gru',
        type=str,
        choices=['mlp', 'gru', 'lstm'],
        help='The core architecture of the model',
    )

    parser.add_argument(
        '-s', '--no-info-share',
        dest='no_sharing',
        action='store_true',
        default=False,
        help='If specified, information sharing is disabled between agents.'
    )

    parser.add_argument(
        '-f', '--use-fingerprint',
        dest='use_fingerprint',
        action='store_true',
        default=False,
        help='If specified, fingerprints are added to observations (see https://arxiv.org/abs/1702.08887).'
    )

    args = parser.parse_args()

    # get env
    env = envs.make_mpe_env()

    # register new algorithm
    marl.algos.register_algo(algo_name="iql", style="il", script=scripts.run_iql)

    # initialize algorithm
    iql = marl.algos.iql(hyperparam_source="mpe")
    # iql.algo_parameters = config.mpe['algo_parameters']

    # build agent model based on env + algorithms + user preference if checked available
    model_config = config.mpe['model_preference']
    model_config.update({'core_arch': args.model_arch})
    model = marl.build_model(env, iql, model_preference=model_config)

    # start learning + extra experiment settings if needed. remember to check ray.yaml before use
    gpu_count = torch.cuda.device_count()
    iql.fit(
        env,
        model,
        stop=config.mpe['stop_condition'],
        local_mode=gpu_count == 0,
        num_gpus=gpu_count,
        num_workers=1,
        share_policy='individual',
        checkpoint_freq=10,
        info_sharing=not args.no_sharing,
        use_fingerprint=args.use_fingerprint,
    )
