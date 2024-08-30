import copy
from collections import Counter
from typing import Dict

from gym.spaces import Tuple
from marllib.marl.algos.scripts.coma import restore_model
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from ray import tune
from ray.rllib.agents.dqn import DEFAULT_CONFIG as BQL_Config
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.tune import CLIReporter
from ray.util.ml_utils.dict import merge_dicts

from niql import seed
from niql.algo import BQLTrainer, WBQLPolicy
from niql.envs import DEBUG_ENVS


def before_learn_on_batch(batch: MultiAgentBatch, workers: WorkerSet, config: Dict, *args, **kwargs):
    if "summary_writer" in kwargs:
        summary_writer = kwargs["summary_writer"]
        policy_map = kwargs["policy_map"]
        timestep = list(policy_map.values())[0].global_timestep
        for policy_id, agent_batch in batch.policy_batches.items():
            policy = policy_map[policy_id]
            setattr(policy, "summary_writer", summary_writer)

            if config.get("env_name") in DEBUG_ENVS:
                summary_writer.add_histogram(policy_id + "/reward_dist2", agent_batch[SampleBatch.REWARDS], timestep)
                stats = Counter(agent_batch[SampleBatch.REWARDS])
                summary_writer.add_scalars(policy_id + "/reward_dist", {str(k): v for k, v in stats.items()}, timestep)

        # if config.get("env_name") in DEBUG_ENVS and "replay_buffer" in kwargs:
        #     replay_buffer = kwargs["replay_buffer"]
        #     replay_buffer.plot_statistics(summary_writer, timestep)

        summary_writer.flush()
    return batch


def run_bql(model_class, exp, run_config, env, stop, restore):
    model_name = "BQL_Model"
    ModelCatalog.register_custom_model(model_name, model_class)

    _param = AlgVar(exp)

    algorithm = exp["algorithm"]
    episode_limit = env["episode_limit"]
    train_batch_episode = _param["batch_episode"]
    lr = _param["lr"]
    lr_schedule = _param.get("lr_schedule")
    buffer_size = _param["buffer_size"]
    target_network_update_frequency = _param["target_network_update_freq"]
    final_epsilon = _param["final_epsilon"]
    epsilon_timesteps = _param["epsilon_timesteps"]
    reward_standardize = _param["reward_standardize"]
    optimizer = _param["optimizer"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    back_up_config["num_agents"] = 1  # one agent one model IQL

    BQL_Config.update(
        {
            "rollout_fragment_length": 1,
            "batch_mode": "complete_episodes",  # "truncate_episodes",
            "buffer_size": buffer_size * episode_limit,  # in timesteps
            "train_batch_size": train_batch_episode,  # in sequence
            "target_network_update_freq": episode_limit * target_network_update_frequency,  # in timesteps
            "learning_starts": episode_limit * train_batch_episode,
            "lr": lr if restore is None else 1e-10,
            "lr_schedule": lr_schedule,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            },
            "similarity_threshold": _param.get("similarity_threshold"),
            "comm_dim": _param.get("comm_dim", 0),
            "comm_aggregator_dims": _param.get("comm_aggregator_dims"),
            "tdw_schedule": _param.get("tdw_schedule"),
            "comm_num_agents": env["num_agents"],
        })

    BQL_Config["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    BQL_Config["optimizer"] = optimizer
    BQL_Config["training_intensity"] = None
    BQL_Config['before_learn_on_batch'] = before_learn_on_batch
    BQL_Config["info_sharing"] = exp["info_sharing"]
    BQL_Config["use_fingerprint"] = exp["use_fingerprint"]
    space_obs = env["space_obs"]["obs"]
    setattr(space_obs, 'original_space', copy.deepcopy(space_obs))
    BQL_Config["obs_space"] = space_obs
    action_space = env["space_act"]
    BQL_Config["act_space"] = Tuple([action_space])
    BQL_Config["gamma"] = _param.get("gamma", BQL_Config["gamma"])
    BQL_Config["callbacks"] = _param.get("callbacks", BQL_Config["callbacks"])
    BQL_Config["lambda"] = _param["lambda"]
    BQL_Config["tau"] = _param["tau"]
    BQL_Config["beta"] = _param["beta"]
    BQL_Config["sharing_batch_size"] = _param["sharing_batch_size"]
    BQL_Config["algorithm"] = algorithm
    BQL_Config["env_name"] = exp["env"]
    BQL_Config["enable_stochastic_eviction"] = _param.get("enable_stochastic_eviction", False)

    # create trainer
    trainer = BQLTrainer.with_updates(
        name=algorithm.upper(),
        default_config=BQL_Config,
    )
    if algorithm.lower() in ["wbql"]:
        trainer = trainer.with_updates(
            get_policy_class=lambda c: WBQLPolicy,
        )

    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    param_sharing_suffix = []
    if exp['model_arch_args']['model'] == 'MatrixGameSplitQMLP':
        param_sharing_suffix = ["ns"]
    comm_suffix = []
    if _param.get("comm_dim", 0) > 0:
        comm_suffix = ["comm"]
    running_name = '_'.join([algorithm, arch, map_name] + comm_suffix + param_sharing_suffix)
    model_path = restore_model(restore, exp)

    # config update
    # set policy IDs
    for policy_id, (_, obs_space, act_space, conf) in run_config["multiagent"]["policies"].items():
        conf["policy_id"] = policy_id
        run_config["multiagent"][policy_id] = (_, obs_space, act_space, conf)

    # add observation function
    # config["multiagent"]["observation_fn"] = ObservationCommWrapper(config["multiagent"]["policy_mapping_fn"])
    run_config.update({
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": back_up_config,
            "fcnet_hiddens": back_up_config["model_arch_args"]["hidden_layer_dims"],
            "fcnet_activation": back_up_config["model_arch_args"]["fcnet_activation"],
            "custom_model": model_name,
        },
        "evaluation_interval": 10,  # x timesteps_per_iteration (default is 1000)
        "evaluation_num_episodes": 10,
        "evaluation_config": {
            "explore": False,
        },
        "seed": seed,
        "num_workers": 0,
        "batch_mode": "complete_episodes",
    })

    results = tune.run(
        trainer,
        name=running_name,
        checkpoint_at_end=exp['checkpoint_end'],
        checkpoint_freq=exp['checkpoint_freq'],
        restore=model_path,
        stop=stop,
        config=run_config,
        verbose=1,
        progress_reporter=CLIReporter(),
        local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"],
    )

    return results
