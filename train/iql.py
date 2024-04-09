import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import ray
import torch
from marllib import marl

from niql import envs, scripts, config, utils
from niql.models import *  # noqa

os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
seed = 321
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '-m', '--model_arch',
        default='mlp',
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

    mode = 'eval'

    # get env
    env = envs.make_matrix_game_env()

    exp_config = config.COOP_MATRIX
    gpu_count = torch.cuda.device_count()

    # register new algorithm
    marl.algos.register_algo(
        algo_name="iql",
        style="il",
        script=scripts.run_iql if mode == 'train' else utils.load_iql_checkpoint,
    )

    # initialize algorithm
    iql = marl.algos.iql(hyperparam_source="mpe")
    iql.algo_parameters = exp_config['algo_parameters']

    # build agent model based on env + algorithms + user preference if checked available
    model_config = exp_config['model_preference']
    model_config.update({'core_arch': args.model_arch})
    model = marl.build_model(env, iql, model_preference=exp_config['model_preference'])
    if model_config.get('custom_model'):
        model = (eval(model_config['custom_model']), model[1])

    if mode == 'train':
        # start learning + extra experiment settings if needed. remember to check ray.yaml before use
        iql.fit(
            env,
            model,
            stop=exp_config['stop_condition'],
            local_mode=gpu_count == 0,
            num_gpus=gpu_count,
            num_workers=1,
            share_policy='individual',
            checkpoint_freq=10,
            info_sharing=not args.no_sharing,
            use_fingerprint=args.use_fingerprint,
        )
    else:
        base = 'exp_results/iql_mlp_all_scenario/IQL_CoopMatrixGame_all_scenario_8c0d4_00000_0_2024-04-09_20-40-13'
        restore_path = {
            'params_path': f'{base}/params.json',  # experiment configuration
            'model_path': f'{base}/checkpoint_000010/checkpoint-10',  # checkpoint path
        }

        # register new algorithm
        marl.algos.register_algo(
            algo_name="iql",
            style="il",
            script=scripts.run_iql if mode == 'train' else utils.load_iql_checkpoint,
        )

        results = iql.fit(
            env,
            model,
            stop=exp_config['stop_condition'],
            local_mode=gpu_count == 0,
            num_gpus=gpu_count,
            num_workers=1,
            share_policy='individual',
            checkpoint_freq=10,
            info_sharing=not args.no_sharing,
            use_fingerprint=args.use_fingerprint,
            restore_path=restore_path
        )

        ray.init(local_mode=True, num_gpus=gpu_count)
        agent, pmap = results.trainer, results.pmap
        env_instance, env_info = env

        # Inference
        for _ in range(1):
            obs = env_instance.reset()
            done = {"__all__": False}
            states = {
                actor_id: agent.get_policy(pmap(actor_id)).get_initial_state()
                for actor_id in obs
            }

            step = 0
            with torch.no_grad():
                while not done["__all__"]:
                    action_dict = {}
                    for agent_id in obs.keys():
                        policy = agent.get_policy(pmap(agent_id))
                        agent_obs = [[0, 0, 0]] if step == 0 else [[0, 0, 1]]  # obs[agent_id]["obs"]
                        action_dict[agent_id], states[agent_id], info = policy.compute_single_action(
                            agent_obs,
                            states[agent_id],
                            explore=False,
                        )
                        print(f'state={step}, agent={agent_id}, q-values={info["q-values"]}')

                    obs, reward, done, info = env_instance.step(action_dict)
                    step += 1

        env_instance.close()
        ray.shutdown()
        logger.info("Inference finished!")
