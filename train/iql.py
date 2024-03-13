import torch
from marllib import marl
from niql import env, scripts, config

if __name__ == '__main__':
    # get env
    env = env.make_mpe_env()

    # register new algorithm
    marl.algos.register_algo(algo_name="iql2", style="il", script=scripts.run_iql)

    # initialize algorithm
    iql = marl.algos.iql2  # (hyperparam_source="mpe")
    iql.algo_parameters = config.mpe['algo_parameters']

    # build agent model based on env + algorithms + user preference if checked available
    model = marl.build_model(env, iql, model_preference=config.mpe['model_preference'])

    # start learning + extra experiment settings if needed. remember to check ray.yaml before use
    iql.fit(
        env,
        model,
        stop=config.mpe['stop_condition'],
        local_mode=True,
        num_gpus=torch.cuda.device_count(),
        num_workers=1,
        share_policy='individual',
        checkpoint_freq=10,
    )
