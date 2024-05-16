import copy
from typing import Callable, Any, Dict

import torch
from gym.spaces import Tuple as GymTuple
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.algos.core.VD.iql_vdn_qmix import JointQTrainer
from marllib.marl.algos.scripts.coma import restore_model
from marllib.marl.algos.utils.setup_utils import AlgVar
from ray import tune
from ray.rllib.agents import Trainer
from ray.rllib.agents.dqn import DEFAULT_CONFIG as IQL_Config
from ray.rllib.agents.qmix import DEFAULT_CONFIG as JointQ_Config
from ray.rllib.agents.qmix.qmix_policy import _mac
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.tune import register_env
from ray.util.ml_utils.dict import merge_dicts

from niql.algo import IMIXTrainer, BQLTrainer, BQLPolicy, IQLTrainer
from niql.algo.vdn_qmix import JointQPolicy
from niql.envs.wrappers import create_fingerprint_env_wrapper_class
from niql.execution_plans import joint_episode_execution_plan
from niql.utils import to_numpy


def determine_multiagent_policy_mapping(exp_info, env_info):
    map_name = exp_info["env_args"]["map_name"]
    agent_name_ls = env_info["agent_name_ls"]
    policy_mapping_info = env_info["policy_mapping_info"][map_name]
    shared_policy_name = "default_policy" if exp_info["agent_level_batch_update"] else "shared_policy"
    if exp_info["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared".format(map_name))

        policies = {shared_policy_name}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: shared_policy_name)

    elif exp_info["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
            policies = {"shared_policy"}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: "shared_policy")
        else:
            policies = {
                "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif exp_info["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(exp_info["share_policy"]))

    return policies, policy_mapping_fn


class Checkpoint:
    def __init__(self, env_name: str, map_name: str, trainer: Trainer, pmap: Callable):
        self.env_name = env_name
        self.map_name = map_name
        self.trainer = trainer
        self.pmap = pmap


class NullLogger:
    """Logger for RLlib to disable logging"""

    def __init__(self, config=None):
        self.config = config
        self.logdir = ""

    def _init(self):
        pass

    def on_result(self, result):
        pass

    def update_config(self, config):
        pass

    def close(self):
        pass

    def flush(self):
        pass


def load_iql_checkpoint(model_class, exp, run_config, env, stop, restore) -> Checkpoint:
    """
    Helper function to retrieve the ray trainer from which the actual policies are accessed.
    """
    model_name = "IQL_Q_Model"
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

    if run_config["multiagent"] is not None:
        run_config["multiagent"].update({
            "policy_map_capacity": 100,
            "policy_map_cache": None,
            "policies_to_train": None,
            "observation_fn": None,
            "replay_mode": "independent",
            "count_steps_by": "env_steps",
        })

    back_up_config["num_agents"] = 1  # one agent one model IQL
    model_config = {
        "max_seq_len": episode_limit,  # dynamic
        "custom_model_config": back_up_config,
        "custom_model": model_name,
        "lstm_cell_size": None,
        "fcnet_activation": back_up_config["model_arch_args"]["fcnet_activation"],
        "fcnet_hiddens": back_up_config["model_arch_args"]["hidden_layer_dims"]
    }

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
            "mixer": exp["algo_args"]["mixer"] if algorithm == 'imix' else algorithm,
            "max_neighbours": env["num_agents"] - 1,
            "comm_dim": _param.get("comm_dim", 0)
        })

    IQL_Config["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    IQL_Config["optimizer"] = optimizer
    IQL_Config["training_intensity"] = None
    # JointQ_Config['before_learn_on_batch'] = before_learn_on_batch
    IQL_Config["info_sharing"] = exp["info_sharing"]
    IQL_Config["use_fingerprint"] = exp["use_fingerprint"]
    space_obs = env["space_obs"]["obs"]
    setattr(space_obs, 'original_space', copy.deepcopy(space_obs))
    IQL_Config["obs_space"] = space_obs
    action_space = env["space_act"]
    IQL_Config["act_space"] = GymTuple([action_space])
    IQL_Config["lambda"] = _param["lambda"]
    IQL_Config["tau"] = _param["tau"]
    IQL_Config["enable_joint_buffer"] = _param.get("enable_joint_buffer")
    IQL_Config["sharing_batch_size"] = _param["sharing_batch_size"]
    IQL_Config["beta"] = _param["beta"]
    IQL_Config["reconcile_rewards"] = _param.get("reconcile_rewards")
    IQL_Config["use_obs_encoder"] = _param.get("use_obs_encoder")
    if "gamma" in _param:
        IQL_Config["gamma"] = _param["gamma"]

    # create trainer
    IQL_Config.update(run_config)
    IQL_Config["model"].update(model_config)
    model_path = restore_model(restore, exp)
    if algorithm == 'imix':
        trainer_class = IMIXTrainer.with_updates(
            name=algorithm.upper(),
            default_config=IQL_Config,
        )
    elif algorithm == 'bql':
        trainer_class = BQLTrainer.with_updates(
            get_policy_class=lambda c: BQLPolicy,
            execution_plan=joint_episode_execution_plan,
        )
    elif algorithm == 'dbql':
        trainer_class = BQLTrainer.with_updates(
            name=algorithm.upper(),
            default_config=IQL_Config,
        )
    else:
        trainer_class = IQLTrainer.with_updates(
            name=algorithm.upper(),
            default_config=IQL_Config,
        )
    trainer = trainer_class(config=IQL_Config, logger_creator=lambda c: NullLogger(c))
    trainer.restore(model_path)

    map_name = exp["env_args"]["map_name"]
    pmap = run_config["multiagent"]["policy_mapping_fn"]
    chkpt = Checkpoint(run_config["env"], map_name, trainer, pmap)
    return chkpt


def load_joint_q_checkpoint(model: Any, exp: Dict, run: Dict, env: Dict,
                            stop: Dict, restore: Dict) -> Checkpoint:
    """ This retrieves the trainer for the IQL, VDN, and QMIX algorithm.
    Adapted from Marllib.

    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """
    model_name = "Joint_Q_Model"
    ModelCatalog.register_custom_model(model_name, model)

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

    if run["multiagent"] is not None:
        run["multiagent"].update({
            "policy_map_capacity": 100,
            "policy_map_cache": None,
            "policies_to_train": None,
            "observation_fn": None,
            "replay_mode": "independent",
            "count_steps_by": "env_steps",
        })

    mixer_dict = {
        "qmix": "qmix",
        "vdn": "vdn",
        "iql": None
    }

    config = {
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": back_up_config,
            "custom_model": model_name,
            "lstm_cell_size": None,
            "fcnet_activation": back_up_config["model_arch_args"]["fcnet_activation"],
            "fcnet_hiddens": back_up_config["model_arch_args"]["hidden_layer_dims"]
        },
    }

    config.update(run)

    JointQ_Config.update(
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
            "mixer": mixer_dict[algorithm]
        })

    JointQ_Config["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    JointQ_Config["optimizer"] = optimizer
    JointQ_Config["training_intensity"] = None
    JointQ_Config["gamma"] = _param.get("gamma", JointQ_Config["gamma"])
    JointQ_Config["callbacks"] = _param.get("callbacks", JointQ_Config["callbacks"])
    JointQ_Config["env"] = run["env"]
    JointQ_Config.update(config)

    JQTrainer = JointQTrainer.with_updates(
        name=algorithm.upper(),
        default_policy=JointQPolicy,
        default_config=JointQ_Config
    )
    trainer = JQTrainer(config=JointQ_Config, logger_creator=lambda c: NullLogger(c))
    model_path = restore_model(restore, exp)
    trainer.restore(model_path)

    map_name = exp["env_args"]["map_name"]
    pmap = run["multiagent"]["policy_mapping_fn"]

    chkpt = Checkpoint(run["env"], map_name, trainer, pmap)

    return chkpt


def vdn_qmix_custom_compute_actions(policy,
                                    obs_batch,
                                    state_batches=None,
                                    prev_action_batch=None,
                                    prev_reward_batch=None,
                                    info_batch=None,
                                    episodes=None,
                                    explore=None,
                                    timestep=None,
                                    **kwargs):
    explore = explore if explore is not None else policy.config["explore"]
    obs_batch, action_mask, _ = policy._unpack_observation(obs_batch)
    # We need to ensure we do not use the env global state
    # to compute actions

    # Compute actions
    with torch.no_grad():
        obs_batch = torch.as_tensor(obs_batch, dtype=torch.float, device=policy.device)
        q_values, hiddens = _mac(
            policy.model,
            obs_batch, [
                torch.as_tensor(
                    np.array(s), dtype=torch.float, device=policy.device)
                for s in state_batches
            ])
        avail = torch.as_tensor(action_mask, dtype=torch.float, device=policy.device)
        masked_q_values = q_values.clone()
        masked_q_values[avail == 0.0] = -float("inf")
        masked_q_values_folded = torch.reshape(masked_q_values, [-1] + list(masked_q_values.shape)[2:])
        actions, _ = policy.exploration.get_exploration_action(
            action_distribution=TorchCategorical(masked_q_values_folded),
            timestep=timestep,
            explore=explore)
        actions = torch.reshape(
            actions,
            list(masked_q_values.shape)[:-1]).cpu().numpy()
        hiddens = [s.cpu().numpy() for s in hiddens]

        # compute q-value through mixer
        state = obs_batch
        agent_1_q_val = torch.squeeze(q_values[:, 0, :])
        agent_2_q_val = torch.squeeze(q_values[:, 1, :])

        if policy.mixer:
            agent_1_q = agent_1_q_val[0]  # agent 1 action
            agent_2_q = agent_2_q_val[1]  # agent 2 action
            chosen_action_qvals = torch.tensor([agent_1_q, agent_2_q], dtype=torch.float)
            chosen_action_qvals = chosen_action_qvals.view(-1, 1, 2)
            q_tot = policy.mixer(chosen_action_qvals, state)

            info = {
                'agent_1_q_val': [to_numpy(agent_1_q_val).tolist()],
                'agent_2_q_val': [to_numpy(agent_2_q_val).tolist()],
                'q-values': [[agent_1_q.item(), agent_2_q.item()]],
                'q_tot': [q_tot.squeeze().item() * 2],
            }
        else:
            info = {
                'agent_1_q_val': [to_numpy(agent_1_q_val).tolist()],
                'agent_2_q_val': [to_numpy(agent_2_q_val).tolist()],
            }
    return tuple(actions.transpose([1, 0])), hiddens, info
