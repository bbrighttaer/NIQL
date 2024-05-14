"""
Adapted from rllib
"""

import logging

import gym
import numpy as np
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from niql.models.slim_fc import SlimFC

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class DuelingQFCN(TorchModelV2, nn.Module):
    """Generic dueling fully connected network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")

        layers = []
        prev_layer_size = int(np.product(obs_space.shape)) + model_config.get("comm_dim", 0)
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    batch_norm=True,
                )
            )
            prev_layer_size = size

        # remaining hidden layer
        if len(hiddens) > 0:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=hiddens[-1],
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    batch_norm=True,
                )
            )
            prev_layer_size = hiddens[-1]

        self.base_model = nn.Sequential(*layers)

        self.advantage_layer = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self.value_layer = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        x = self.base_model(obs)
        advantages = self.advantage_layer(x)
        values = self.value_layer(x)
        q_values = values + (advantages - advantages.mean())
        return q_values, state
