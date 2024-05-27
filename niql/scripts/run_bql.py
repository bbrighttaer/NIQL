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

from niql.algo import BQLTrainer, BQLPolicy
from niql.callbacks import ObservationCommWrapper
from niql.envs import DEBUG_ENVS
from niql.execution_plans import joint_episode_execution_plan


def before_learn_on_batch(batch: MultiAgentBatch, workers: WorkerSet, config: Dict, *args, **kwargs):
    if "summary_writer" in kwargs:
        summary_writer = kwargs["summary_writer"]
        policy_map = kwargs["policy_map"]
        timestep = list(policy_map.values())[0].global_timestep
        for policy_id, agent_batch in batch.policy_batches.items():
            policy = policy_map[policy_id]
            setattr(policy, "summary_writer", summary_writer)

            if config.get("env_name") in DEBUG_ENVS:
                stats = Counter(agent_batch[SampleBatch.REWARDS])
                summary_writer.add_scalars(policy_id + "/reward_dist_in_batch",
                                           {str(k): v for k, v in stats.items()}, timestep)

        if config.get("env_name") in DEBUG_ENVS and "replay_buffer" in kwargs:
            replay_buffer = kwargs["replay_buffer"]
            replay_buffer.plot_statistics(summary_writer, timestep)

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

    # set policy IDs
    for policy_id, (_, obs_space, act_space, conf) in config["multiagent"]["policies"].items():
        conf["policy_id"] = policy_id
        config["multiagent"][policy_id] = (_, obs_space, act_space, conf)

    # add observation function
    config["multiagent"]["observation_fn"] = ObservationCommWrapper(config["multiagent"]["policy_mapping_fn"])

    BQL_Config.update(
        {
            "rollout_fragment_length": 1,
            "batch_mode": "complete_episodes",  # "truncate_episodes",
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
            "similarity_threshold": _param.get("similarity_threshold"),
            "comm_dim": _param.get("comm_dim", 0)
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
    # BQL_Config["enable_joint_buffer"] = _param["enable_joint_buffer"]
    BQL_Config["sharing_batch_size"] = _param["sharing_batch_size"]
    BQL_Config["algorithm"] = algorithm
    BQL_Config["env_name"] = exp["env"]

    # create trainer
    trainer = BQLTrainer.with_updates(
        name=algorithm.upper(),
        default_config=BQL_Config,
    )
    if algorithm.lower() == 'bql':
        trainer = trainer.with_updates(
            get_policy_class=lambda c: BQLPolicy,
        )
    if _param["enable_joint_buffer"]:
        trainer = trainer.with_updates(
            execution_plan=joint_episode_execution_plan,
        )

    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    param_sharing = 'ns' if exp['model_arch_args'].get('custom_model') == 'MatrixGameSplitQMLP' else ''
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
