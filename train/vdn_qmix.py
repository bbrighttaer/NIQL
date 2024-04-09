import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import ray
import torch
from marllib import marl
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from niql.models import *  # noqa

from niql import envs, config, utils, scripts

logger = logging.getLogger(__name__)

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
    parser.add_argument(
        '-m', '--model_arch',
        default='mlp',
        type=str,
        choices=['mlp', 'gru', 'lstm'],
        help='The core architecture of the model',
    )
    args = parser.parse_args()

    mode = 'eval'

    # get env
    env = envs.make_matrix_game_env()

    # initialise algorithm with hyperparameters
    if args.algo == 'qmix':
        algo = marl.algos.qmix(hyperparam_source="mpe")
    else:
        algo = marl.algos.vdn(hyperparam_source="mpe")
    exp_config = config.COOP_MATRIX
    algo.algo_parameters = exp_config['algo_parameters']

    # build model
    model_config = exp_config['model_preference']
    model_config.update({'core_arch': args.model_arch})
    model = marl.build_model(env, algo, model_preference=exp_config['model_preference'])
    if model_config.get('custom_model'):
        model = (eval(model_config['custom_model']), model[1])

    gpu_count = torch.cuda.device_count()

    if mode == 'train':
        # register execution script
        marl.algos.register_algo(
            algo_name=algo.name,
            style=algo.algo_type,
            script=scripts.run_joint_q
        )

        # start training
        algo.fit(
            env,
            model,
            stop=exp_config['stop_condition'],
            local_mode=gpu_count == 0,
            num_gpus=gpu_count,
            num_workers=0,
            share_policy='all',
            checkpoint_freq=10,
        )
    else:
        base = 'exp_results/vdn_mlp_all_scenario/VDN_grouped_CoopMatrixGame_all_scenario_c2351_00000_0_2024-04-09_20-27-25'
        restore_path = {
            'params_path': f'{base}/params.json',  # experiment configuration
            'model_path': f'{base}/checkpoint_000010/checkpoint-10',  # checkpoint path
        }

        # register execution script
        marl.algos.register_algo(
            algo_name=algo.name,
            style=algo.algo_type,
            script=utils.load_joint_q_checkpoint,
        )

        # start training
        results = algo.fit(
            env,
            model,
            stop=exp_config['stop_condition'],
            local_mode=gpu_count == 0,
            num_gpus=gpu_count,
            num_workers=0,
            share_policy='all',
            checkpoint_freq=10,
            restore_path=restore_path,
        )

        ray.init(local_mode=True, num_gpus=gpu_count)
        agent, pmap = results.trainer, results.pmap
        env_instance, env_info = env

        # Inference
        for _ in range(1):
            obs = env_instance.reset()
            done = {"__all__": False}
            states = {
                actor_id: agent.get_policy(DEFAULT_POLICY_ID).get_initial_state()
                for actor_id in obs
            }

            step = 0
            with torch.no_grad():
                while not done["__all__"]:
                    action_dict = {}
                    agent_obs = []
                    for agent_id in obs.keys():
                        policy = agent.get_policy(DEFAULT_POLICY_ID)
                        agent_obs += [0, 0, 0] if step == 0 else [0, 0, 1]  # obs[agent_id]["obs"]
                    action_dict[agent_id], states[agent_id], info = policy.compute_single_action(
                        np.array(agent_obs).reshape(1, -1),
                        list(states.values()),
                        explore=False,
                    )
                    print(f'state={step}, agent={agent_id}, q-values={info["q-values"]}')

                    obs, reward, done, info = env_instance.step(action_dict)
                    step += 1

        env_instance.close()
        ray.shutdown()
        logger.info("Inference finished!")
