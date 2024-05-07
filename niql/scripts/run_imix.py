import copy
from typing import Dict

from gym.spaces import Tuple
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.algos.scripts.coma import restore_model
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from ray import tune
from ray.rllib.agents.dqn import DEFAULT_CONFIG as IQL_Config
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.tune import CLIReporter, register_env
from ray.util.ml_utils.dict import merge_dicts

from niql.algo import IMIXTrainer
from niql.envs.wrappers import create_fingerprint_env_wrapper_class
from niql.trainer_loaders import determine_multiagent_policy_mapping


def before_learn_on_batch(batch: MultiAgentBatch, workers: WorkerSet, config: Dict, policy_map: dict):
    for policy_id, agent_batch in batch.policy_batches.items():
        neighbor_policies = {p_id: p for p_id, p in policy_map.items() if p_id != policy_id}
        agent_batch['neighbour_policies'] = neighbor_policies
    return batch


def run_imix(model_class, exp, run_config, env, stop, restore):
    model_name = "IMIX_Model"
    ModelCatalog.register_custom_model(model_name, model_class)

    if exp["use_fingerprint"]:
        # new environment name
        env_reg_name = "fp_" + run_config["env"]
        run_config["env"] = env_reg_name

        def create_env(*arg, **kwargs):
            env_class = ENV_REGISTRY.get(exp["env"]) or COOP_ENV_REGISTRY[exp["env"]]
            return create_fingerprint_env_wrapper_class(env_class)(exp["env_args"])

        # add wrapped environment to envs list
        register_env(env_reg_name, create_env)

        # get new environment information
        wrapped_env = create_env()
        env_info = wrapped_env.get_env_info()
        wrapped_env.close()

        # update env information
        env.update(env_info)

        # update multi-agent policy mapping so observation space can match new updated environment info
        policies, policy_mapping_fn = determine_multiagent_policy_mapping(exp, env)
        run_config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        }

    _param = AlgVar(exp)

    algorithm = exp["algorithm"]
    episode_limit = env["episode_limit"]
    train_batch_episode = _param["batch_episode"]
    lr = _param["lr"]
    buffer_size = _param["buffer_size"]
    target_network_update_frequency = _param["target_network_update_freq"]
    final_epsilon = _param["final_epsilon"]
    epsilon_timesteps = _param["epsilon_timesteps"]
    reward_standardize = _param["reward_standardize"]
    optimizer = _param["optimizer"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    back_up_config["num_agents"] = 1  # one agent one model IQL
    config = {
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": back_up_config,
            "custom_model": model_name,
            "fcnet_activation": back_up_config["model_arch_args"]["fcnet_activation"],
            "fcnet_hiddens": back_up_config["model_arch_args"]["hidden_layer_dims"]
        },
    }

    config.update(run_config)

    # add observation function
    # config["multiagent"]["observation_fn"] = ObservationCommWrapper(config["multiagent"]["policy_mapping_fn"])

    IQL_Config.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": buffer_size * episode_limit,  # in timesteps
            "train_batch_size": train_batch_episode,  # in sequence
            "target_network_update_freq": episode_limit * target_network_update_frequency,  # in timesteps
            "learning_starts": episode_limit * train_batch_episode,
            "lr": lr if restore is None else 1e-10,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            },
            "mixer": _param['mixer'],
            "max_neighbours": env["num_agents"] - 1,
        })

    IQL_Config["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    IQL_Config["optimizer"] = optimizer
    IQL_Config["training_intensity"] = None
    IQL_Config['before_learn_on_batch'] = before_learn_on_batch
    IQL_Config["info_sharing"] = exp["info_sharing"]
    IQL_Config["use_fingerprint"] = exp["use_fingerprint"]
    space_obs = env["space_obs"]["obs"]
    setattr(space_obs, 'original_space', copy.deepcopy(space_obs))
    IQL_Config["obs_space"] = space_obs
    action_space = env["space_act"]
    IQL_Config["act_space"] = Tuple([action_space])
    IQL_Config["gamma"] = _param.get("gamma", IQL_Config["gamma"])
    IQL_Config["callbacks"] = _param.get("callbacks", IQL_Config["callbacks"])

    # create trainer
    trainer = IMIXTrainer.with_updates(
        name=algorithm.upper(),
        default_config=IQL_Config,
    )

    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    param_sharing = 'ns' if exp['model_arch_args']['custom_model'] == 'MatrixGameSplitQMLP' else ''
    running_name = '_'.join([algorithm, arch, map_name] + ([param_sharing] if param_sharing else []))
    model_path = restore_model(restore, exp)

    results = tune.run(
        trainer,
        name=running_name,
        checkpoint_at_end=exp['checkpoint_end'],
        checkpoint_freq=exp['checkpoint_freq'],
        restore=model_path,
        stop=stop,
        config=config,
        verbose=1,
        progress_reporter=CLIReporter(),
        local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"],
    )

    return results
