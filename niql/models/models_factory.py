from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from niql.models import DRQNModel


class DQNModelsFactory:

    def __init__(self, agent_obs_space, action_space, config, core_arch):
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.params += list(self.model.parameters())


class BQLModelsFactory:

    def __init__(self, agent_obs_space, action_space, config, core_arch):
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.auxiliary_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.auxiliary_model_target = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else DRQNModel
        ).to(self.device)

        self.params += list(self.model.parameters()) + list(self.auxiliary_model.parameters())
