from marllib import marl

from niql.scripts import run_iql


if __name__ == '__main__':
    # choose environment + scenario
    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

    # register new algorithm
    marl.algos.register_algo(algo_name="iql2", style="il", script=run_iql)

    # initialize algorithm
    iql = marl.algos.iql2  # (hyperparam_source="mpe")
    iql.algo_parameters = {
        'algo_args': {
            'batch_episode': 128,
            'lr': 0.0005,
            'rollout_fragment_length': 1,
            'buffer_size': 5000,
            'target_network_update_freq': 200,
            'final_epsilon': 0.05,
            'epsilon_timesteps': 50000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': True,
        }
    }

    # build agent model based on env + algorithms + user preference if checked available
    model = marl.build_model(env, iql, {"core_arch": "mlp", "encode_layer": "128-256"})

    # start learning + extra experiment settings if needed. remember to check ray.yaml before use
    iql.fit(
        env,
        model,
        stop={
            'episode_reward_mean': 2000,
            'timesteps_total': 10000000,
        },
        local_mode=True,
        num_gpus=0,
        num_workers=0,
        share_policy='individual',
        checkpoint_freq=10,
    )
