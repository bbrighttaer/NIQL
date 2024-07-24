import logging
import os
import random
from argparse import ArgumentParser
from functools import partial

import numpy as np
import ray
import torch
from marllib import marl
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from niql import envs, utils, trainer_loaders, scripts, seed
from niql.config import MODEL_CHECKPOINT_FREQ
from niql.models import *  # noqa

logger = logging.getLogger(__name__)

os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-a', '--algo',
        type=str,
        default='vdn',
        choices=['vdn', 'qmix', 'iql'],
        help='Select which CTDE algorithm to run.',
    )
    # parser.add_argument(
    #     '-m', '--model_arch',
    #     default='mlp',
    #     type=str,
    #     choices=['mlp', 'gru', 'lstm'],
    #     help='The core architecture of the model',
    # )
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
    env, exp_config = envs.get_active_env()

    # register execution script
    marl.algos.register_algo(
        algo_name=args.algo,
        style="VD",
        script=scripts.run_joint_q if mode == 'train' else trainer_loaders.load_joint_q_checkpoint,
    )

    # initialise algorithm with hyperparameters
    algo = getattr(marl.algos, args.algo)
    algo.algo_parameters = exp_config['algo_parameters']



    # build model
    model_config = exp_config['model_preference']
    # model_config.update({'core_arch': args.model_arch})
    model = marl.build_model(env, algo, model_preference=exp_config['model_preference'])
    if model_config.get('model'):
        model = (eval(model_config['model']), model[1])

    gpu_count = torch.cuda.device_count()

    if mode == 'train':
        # start training
        algo.fit(
            env,
            model,
            stop=exp_config['stop_condition'],
            local_mode=gpu_count == 0,
            num_gpus=gpu_count,
            num_workers=5,
            share_policy='all',
            checkpoint_freq=MODEL_CHECKPOINT_FREQ,
        )
    else:
        base = 'exp_results/qmix_mlp_all_scenario/QMIX_grouped_TwoStepsCoopMatrixGame_all_scenario_af63e_00000_0_2024-04-26_10-19-45'
        restore_path = {
            'params_path': f'{base}/params.json',  # experiment configuration
            'model_path': f'{base}/checkpoint_000010/checkpoint-10',  # checkpoint path
        }

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
        policy = agent.get_policy(DEFAULT_POLICY_ID)
        # override the compute actions method
        policy.compute_actions = partial(utils.vdn_qmix_custom_compute_actions, policy)

        # Inference
        for _ in range(1):
            obs = env_instance.reset()
            done = {"__all__": False}
            states = [policy.get_initial_state() for actor_id in obs]

            step = 0
            with torch.no_grad():
                while not done["__all__"]:
                    agent_obs = [0, 0, 1] * 2
                    actions, states, info = policy.compute_single_action(
                        np.array(agent_obs).reshape(1, -1),
                        states,
                        explore=False,
                    )
                    print(f'state={step}, info={info}, agent_obs={agent_obs}')
                    action_dict = {agt_id: action for agt_id, action in zip(obs, actions)}
                    obs, reward, done, info = env_instance.step(action_dict)
                    step += 1

        env_instance.close()
        ray.shutdown()
        logger.info("Inference finished!")
