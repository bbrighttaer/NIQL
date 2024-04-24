import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import ray
import torch
from marllib import marl

from niql import envs, scripts, config, utils, seed
from niql.models import *  # noqa

os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()

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

    parser.add_argument(
        '-e', '--exec_mode',
        default='train',
        type=str,
        choices=['train', 'eval', 'render'],
        help='Execution mode',
    )

    args = parser.parse_args()

    mode = args.exec_mode

    # get env
    env = envs.make_two_step_matrix_game_env()

    exp_config = config.COOP_MATRIX
    gpu_count = torch.cuda.device_count()

    # register new algorithm
    marl.algos.register_algo(
        algo_name="imix",
        style="il",
        script=scripts.run_imix if mode == 'train' else utils.load_iql_checkpoint,
    )

    # initialize algorithm
    imix = marl.algos.imix  # (hyperparam_source="mpe")
    imix.algo_parameters = exp_config['algo_parameters']
    imix.algo_parameters['algo_args']['mixer'] = 'qmix'

    # build agent model based on env + algorithms + user preference if checked available
    model_config = exp_config['model_preference']
    model = marl.build_model(env, imix, model_preference=exp_config['model_preference'])
    if model_config.get('model'):
        model = (eval(model_config['model']), model[1])

    if mode == 'train':
        # start learning + extra experiment settings if needed. remember to check ray.yaml before use
        imix.fit(
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
        base = 'exp_results/imix_mlp_all_scenario/IMIX_CoopMatrixGame_all_scenario_b2116_00000_0_2024-04-20_23-05-43'
        restore_path = {
            'params_path': f'{base}/params.json',  # experiment configuration
            'model_path': f'{base}/checkpoint_000010/checkpoint-10',  # checkpoint path
        }

        results = imix.fit(
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
            neighbour_lookup = {}
            for actor_id in obs:
                for n_id in obs:
                    if actor_id != n_id:
                        neighbour_lookup[actor_id] = n_id

            step = 0
            with torch.no_grad():
                action_dict = {}
                cur_state = np.array([[0, 0, 1]])
                for agent_id in obs.keys():
                    policy = agent.get_policy(pmap(agent_id))
                    n_policy = agent.get_policy(pmap(neighbour_lookup[agent_id]))
                    info = policy.compute_eval_actions(
                        cur_state,
                        states[agent_id],
                        cur_state,
                        n_policy,
                    )
                    print(f'state={step}, agent={agent_id}, q-values={info["q_values"]}, q_tot: {info["q_tot"]}')

                step += 1

        env_instance.close()
        ray.shutdown()
        logger.info("Inference finished!")
