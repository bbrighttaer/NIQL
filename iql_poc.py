import json

from marllib import marl
from marllib.marl import JointQMLP
from marllib.marl.algos.utils.episode_execution_plan import episode_execution_plan
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from ray import tune
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from ray.rllib.agents.qmix import DEFAULT_CONFIG
from ray.tune import CLIReporter
from ray.util.ml_utils.dict import merge_dicts

from policy import IQLPolicy


# create execution script
def run_iql(model_class, exp_info, run_config, env_info, stop_config, restore_config):
    _param = AlgVar(exp_info)

    episode_limit = env_info["episode_limit"]
    train_batch_episode = _param["batch_episode"]
    lr = _param["lr"]
    buffer_size = _param["buffer_size"]
    target_network_update_frequency = _param["target_network_update_freq"]
    final_epsilon = _param["final_epsilon"]
    epsilon_timesteps = _param["epsilon_timesteps"]
    reward_standardize = _param["reward_standardize"]
    optimizer = _param["optimizer"]
    back_up_config = merge_dicts(exp_info, env_info)
    back_up_config.pop("algo_args")  # clean for grid_search

    config = {
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model": JointQMLP,
            "custom_model_config": back_up_config,
        },
    }

    config.update(run_config)

    DEFAULT_CONFIG.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": buffer_size * episode_limit,  # in timesteps
            "train_batch_size": train_batch_episode,  # in sequence
            "target_network_update_freq": episode_limit * target_network_update_frequency,  # in timesteps
            "learning_starts": episode_limit * train_batch_episode,
            "lr": lr if restore_config is None else 1e-10,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            },
            "mixer": None
        })

    DEFAULT_CONFIG["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    DEFAULT_CONFIG["optimizer"] = optimizer
    DEFAULT_CONFIG["training_intensity"] = None

    # create trainer
    IQLTrainer = GenericOffPolicyTrainer.with_updates(
        name="IQLTrainer",
        default_config=DEFAULT_CONFIG,
        default_policy=IQLPolicy,
        get_policy_class=None,
        execution_plan=episode_execution_plan)

    algorithm = exp_info["algorithm"]
    map_name = exp_info["env_args"]["map_name"]
    arch = exp_info["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    if restore_config is not None:
        with open(restore_config["params_path"], 'r') as JSON:
            raw_config = json.load(JSON)
            raw_config = raw_config["model"]["custom_model_config"]['model_arch_args']
            check_config = config["model"]["custom_model_config"]['model_arch_args']
            if check_config != raw_config:
                raise ValueError("is not using the params required by the checkpoint model")
        model_path = restore_config["model_path"]
    else:
        model_path = None

    results = tune.run(
        IQLTrainer,
        name=RUNNING_NAME,
        checkpoint_at_end=exp_info['checkpoint_end'],
        checkpoint_freq=exp_info['checkpoint_freq'],
        restore=model_path,
        stop=stop_config,
        config=config,
        verbose=1,
        progress_reporter=CLIReporter(),
        local_dir=available_local_dir if exp_info["local_dir"] == "" else exp_info["local_dir"],
    )

    return results


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
            'timesteps_total': 100,
        },
        local_mode=True,
        num_gpus=0,
        num_workers=0,
        share_policy='individual',
        checkpoint_freq=10,
    )
