import os
from typing import Tuple, Dict

import yaml
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import dict_update, SYSPARAMs, set_ray
from ray.rllib import MultiAgentEnv
from ray.tune import register_env
from tabulate import tabulate


def make_local_env(
        environment_name: str,
        map_name: str = None,
        force_coop: bool = False,
        abs_path: str = "",
        **env_params
) -> Tuple[MultiAgentEnv, Dict]:
    """
    Adapted from Marllib.
    construct the environment and register.
    Args:
        :param environment_name: name of the environment
        :param map_name: name of the scenario
        :param force_coop: enforce the reward return of the environment to be global
        :param abs_path: env configuration path
        :param env_params: parameters that can be pass to the environment for customizing the environment

    Returns:
        Tuple[MultiAgentEnv, Dict]: env instance & env configuration dict
    """
    # The only change here is to remove the config file loading restrictions
    if abs_path != "":
        env_config_file_path = os.path.join(os.path.dirname(__file__), abs_path)
    else:
        env_config_file_path = os.path.join(os.path.dirname(__file__), "config/{}.yaml".format(environment_name))

    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # update function-fixed config
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], env_params, True)

    # user commandline config
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], user_env_args, True)
    if map_name:
        env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop

    # combine with exp running config
    env_config = set_ray(env_config_dict)

    # initialize env
    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == env_config["env"]:
                check_current_used_env_flag = True

    print(tabulate(env_reg_ls,
                   headers=['Env_Name', 'Check_Status', "Error_Log", "Config_File_Location", "Env_File_Location"],
                   tablefmt='grid'))

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                env_config["env"]))

    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]

    if env_config["force_coop"]:
        register_env(env_reg_name, lambda _: COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"])
    else:
        register_env(env_reg_name, lambda _: ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    return env, env_config
