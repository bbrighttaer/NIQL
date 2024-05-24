import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import ray
import torch
from marllib import marl

import niql.trainer_loaders
from niql import envs, scripts, seed
from niql.config import MODEL_CHECKPOINT_FREQ
from niql.models import *  # noqa

os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
# os.environ['RAY_PICKLE_VERBOSE_DEBUG'] = '1'

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
        help='If specified, information sharing is disabled between agents.'
    )

    parser.add_argument(
        '-f', '--use-fingerprint',
        dest='use_fingerprint',
        action='store_true',
        help='If specified, fingerprints are added to observations (see https://arxiv.org/abs/1702.08887).'
    )

    parser.add_argument(
        '-e', '--exec_mode',
        default='train',
        type=str,
        choices=['train', 'eval', 'render'],
        help='Execution mode',
    )

    parser.add_argument(
        '-d', '--disable_joint_buffer',
        action='store_true',
        help='If specified, separate experience buffers will be used (when using BQL).',
    )

    parser.add_argument(
        '-a', '--algo',
        choices=['bql', 'wbql'],
        default='bql',
        help='The algorithm to use.',
    )

    args = parser.parse_args()

    mode = args.exec_mode

    # get env
    env, exp_config = envs.get_active_env()

    gpu_count = torch.cuda.device_count()

    # register new algorithm
    marl.algos.register_algo(
        algo_name=args.algo,
        style="il",
        script=scripts.run_bql if mode == "train" else niql.trainer_loaders.load_iql_checkpoint,
    )

    # initialize algorithm
    bql = marl.algos.wbql if args.algo == "wbql" else marl.algos.bql  # (hyperparam_source="mpe")
    bql.algo_parameters = exp_config['algo_parameters']
    bql.algo_parameters["algo_args"]["enable_joint_buffer"] = False
    if args.algo == "bql":
        bql.algo_parameters["algo_args"]["enable_joint_buffer"] = not args.disable_joint_buffer

    # build agent model based on env + algorithms + user preference if checked available
    model_config = exp_config['model_preference']
    model = marl.build_model(env, bql, model_preference=model_config)
    if model_config['model']:
        model = (eval(model_config['model']), model[1])

    base = 'exp_results/wbql_gru_all_scenario/WBQL_PredatorPrey_all_scenario_9d20f_00000_0_2024-05-20_10-40-53'
    restore_path = {
        'params_path': f'{base}/params.json',  # experiment configuration
        'model_path': f'{base}/checkpoint_001060/checkpoint-1060',  # checkpoint path
        # 'render': False
    }
    if mode == 'train':
        # start learning + extra experiment settings if needed. remember to check ray.yaml before use
        bql.fit(
            env,
            model,
            stop=exp_config['stop_condition'],
            local_mode=gpu_count == 0,
            num_gpus=gpu_count,
            num_workers=1,
            share_policy='individual',
            checkpoint_freq=MODEL_CHECKPOINT_FREQ,
            info_sharing=not args.no_sharing,
            use_fingerprint=args.use_fingerprint,
            # restore_path=restore_path
        )
    else:
        results = bql.fit(
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
                    cur_state = np.array([0, 0, 0])
                    neighbour_state = cur_state  # np.array([0, 0, 0])
                    for agent_id in obs.keys():
                        policy = agent.get_policy(pmap(agent_id))
                        if hasattr(policy, "neighbour_messages"):
                            n_msg = policy.get_message(neighbour_state.reshape(1, 1, len(cur_state)))
                            policy.neighbour_messages.append(n_msg)
                        agent_obs = cur_state  # obs[agent_id]["obs"]
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
