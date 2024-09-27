# MIT License
from functools import partial
from typing import Any, Dict

from marllib.marl.algos.scripts.coma import restore_model
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from ray import tune
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as JointQ_Config
from ray.rllib.models import ModelCatalog
from ray.tune import CLIReporter
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.utils import merge_dicts

from niql import seed
from niql.algo import JointQPolicy, IQLPolicy, HIQLPolicy, BQLPolicy, WBQLPolicy, WIQLPolicy
from niql.algo.iqlps_vdn_qmix import JointQTrainer
from niql.callbacks import EvaluationCSVLoggerCallback, NIQLCallbacks
from niql.execution_plans import episode_execution_plan


# Copyright (c) 2023 Replicable-MARL
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def get_policy_class(algorithm, config_):
    return {
        "qmix": JointQPolicy,
        "vdn": JointQPolicy,
        "iqlps": JointQPolicy,
        "iql": IQLPolicy,
        "hiql": HIQLPolicy,
        "bql": BQLPolicy,
        "wbql": WBQLPolicy,
        "wiql": WIQLPolicy,
    }.get(algorithm)


def run_experiment(model: Any, exp: Dict, running_config: Dict, env: Dict,
                   stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the IQL, VDN, and QMIX algorithm using Ray RLlib.
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

    ModelCatalog.register_custom_model(
        "Joint_Q_Model", model)

    _param = AlgVar(exp)

    algorithm = exp["algorithm"]
    algo_type = exp["algo_type"]
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

    mixer_dict = {
        "qmix": "qmix",
        "vdn": "vdn",
    }

    JointQ_Config.update(
        {
            "algo_type": algo_type,
            "rollout_fragment_length": 1,
            "normalize_actions": False,
            "buffer_size": buffer_size,  # buffer_size * episode_limit,  # in timesteps
            "train_batch_size": train_batch_episode,  # in sequence
            "target_network_update_freq": target_network_update_frequency,  # in timesteps
            "learning_starts": _param.get("learning_starts", episode_limit * train_batch_episode),  # samples in buffer
            "lr": lr if restore is None else 1e-10,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            },
            "tau": _param["tau"],
            "lambda": _param.get("lambda", 0.2),
            "gamma": _param.get("gamma", 0.99),
            "mixer": mixer_dict.get(algorithm),
            "tdw_eps": _param.get("tdw_eps", 0.1),
            "tdw_schedule": _param.get("tdw_schedule"),
            "add_action_dim": _param.get("add_action_dim", False),
            "soft_target_update": _param.get("soft_target_update", True),
            "reward_standardize": reward_standardize,  # this may affect the final performance if you turn it on
            "optimizer": optimizer,
            "training_intensity": None,
            "batch_mode": "complete_episodes"
        })

    JQTrainer = JointQTrainer.with_updates(
        name=algorithm.upper(),
        default_config=JointQ_Config,
        default_policy=None,
        get_policy_class=partial(get_policy_class, algorithm),
        execution_plan=episode_execution_plan
    )

    # config update
    running_config.update({
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": back_up_config,
            "fcnet_hiddens": back_up_config["model_arch_args"]["hidden_layer_dims"],
            "fcnet_activation": "relu"
        },
        "evaluation_interval": 10,  # x timesteps_per_iteration (default is 1000)
        "evaluation_num_episodes": 20,
        "evaluation_config": {
            "explore": False,
        },
        "seed": seed,
        "num_workers": 0,
        "batch_mode": "complete_episodes",
        "callbacks": NIQLCallbacks
    })

    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name, "joint" if algo_type.lower() == "vd" else algo_type.lower()])
    model_path = restore_model(restore, exp)

    results = tune.run(JQTrainer,
                       name=RUNNING_NAME,
                       checkpoint_at_end=exp['checkpoint_end'],
                       checkpoint_freq=exp['checkpoint_freq'],
                       restore=model_path,
                       stop=stop,
                       config=running_config,
                       verbose=1,
                       callbacks=[EvaluationCSVLoggerCallback()],
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"])

    return results
