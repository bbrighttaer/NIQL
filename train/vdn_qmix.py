import torch
from marllib import marl
from niql import env, config

if __name__ == '__main__':
    # get env
    env = env.make_mpe_env()

    # set algorithm to use
    ctde = getattr(marl.algos, 'qmix')

    # initialise algorithm with hyperparameters
    algo = ctde  # (hyperparam_source='mpe')
    ctde.algo_parameters = config.mpe['algo_parameters']

    # build model
    model = marl.build_model(env, algo, model_preference=config.mpe['model_preference'])

    # start training
    # gpu_count = torch.cuda.device_count()
    # algo.fit(
    #     env,
    #     model,
    #     stop=config.mpe['stop_condition'],
    #     local_mode=gpu_count == 0,
    #     num_gpus=gpu_count,
    #     num_workers=0,
    #     share_policy='all',
    #     checkpoint_freq=10,
    # )

    # rendering
    r_path = 'exp_results/qmix_mlp_simple_spread/QMIX_grouped_mpe_simple_spread_697a8_00000_0_2024-03-13_02-21-34/checkpoint_003850'
    p_path = 'exp_results/qmix_mlp_simple_spread/QMIX_grouped_mpe_simple_spread_697a8_00000_0_2024-03-13_02-21-34/params.json'
    algo.render(env, model,
                restore_path={'params_path': p_path,  # experiment configuration
                              'model_path': r_path,  # checkpoint path
                              'render': True},  # render
                local_mode=True,
                share_policy="all",
                checkpoint_end=False,
                )
