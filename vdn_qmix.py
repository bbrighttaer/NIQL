from marllib import marl
from niql import env, config

if __name__ == '__main__':
    # get env
    env = env.make_mpe_env()

    # set algorithm to use
    ctde = getattr(marl.algos, 'qmix')

    # initialise algorithm with hyper-parameters
    algo = ctde  # (hyperparam_source='mpe')
    ctde.algo_parameters = config.mpe['algo_parameters']

    # build model
    model = marl.build_model(env, algo, model_preference=config.mpe['model_preference'])

    # start training
    algo.fit(
        env,
        model,
        stop={
            'episode_reward_mean': 2000,
            'timesteps_total': 10000000,
        },
        local_mode=True,
        num_gpus=0,
        num_workers=0,
        share_policy='all',
        checkpoint_freq=10,
    )
