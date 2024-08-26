from typing import Tuple, Dict, Any

from marllib.marl import _Algo as _BaseAlgo, recursive_dict_update, POlICY_REGISTRY
from ray.rllib import MultiAgentEnv

from niql.runners.run_cc import run_cc
from niql.runners.run_il import run_il
from niql.runners.run_vd import run_vd


class _Algo(_BaseAlgo):

    def fit(self, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], stop: Dict = None,
            **running_params) -> None:
        """
        Entering point of the whole training
        Args:
            :param env: a tuple of environment instance and environmental configuration
            :param model: a tuple of model class and model configuration
            :param stop: dict of running stop condition
            :param running_params: other configuration to customize the training
        Returns:
            None
        """

        env_instance, info = env
        model_class, model_info = model

        self.config_dict = info
        self.config_dict = recursive_dict_update(self.config_dict, model_info)

        self.config_dict = recursive_dict_update(self.config_dict, self.algo_parameters)
        self.config_dict = recursive_dict_update(self.config_dict, running_params)

        self.config_dict['algorithm'] = self.name

        if self.algo_type == "IL":
            return run_il(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "VD":
            return run_vd(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "CC":
            return run_cc(self.config_dict, env_instance, model_class, stop=stop)
        else:
            raise ValueError("not supported type {}".format(self.algo_type))


class _AlgoManager:
    def __init__(self):
        """An algorithm pool class
        """
        for algo_name in POlICY_REGISTRY:
            setattr(_AlgoManager, algo_name, _Algo(algo_name))

    def register_algo(self, algo_name: str, style: str, script: Any):
        """
        Algorithm registration
        Args:
            :param algo_name: algorithm name
            :param style: algorithm learning style from ["il", "vd", "cc"]
            :param script: a running script to start training
        Returns:
            None
        """
        setattr(_AlgoManager, algo_name, _Algo(algo_name + "_" + style))
        POlICY_REGISTRY[algo_name] = script


algos = _AlgoManager()
