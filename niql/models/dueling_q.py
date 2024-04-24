import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn


class DuelingQModel(TorchModelV2, nn.Module):
    """
    Dueling DQN wrapper for models.
    """

    def __init__(self, input_dim, n_actions, obs_space, action_space, model_config, name, in_activation_fn=None):
        TorchModelV2.__init__(self, obs_space, action_space, n_actions, model_config, name)
        nn.Module.__init__(self)
        self.value_layer = nn.Linear(input_dim, 1)
        self.advantage_layer = nn.Linear(input_dim, n_actions)
        # useful for instances where the wrapped model's output is linear.
        if in_activation_fn:
            self.in_activation_fn = get_activation_fn(in_activation_fn, "torch")
        else:
            self.in_activation_fn = lambda x: x

    def forward(self, input_dict, *args, **kwargs):
        x = input_dict["obs_flat"].float()
        advantages = self.advantage_layer(x)
        values = self.value_layer(x)
        q_values = values + (advantages - advantages.mean())
        return (q_values, values), []
