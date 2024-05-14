"""
Adapted from rllib
"""
from typing import Any

import torch
import torch.nn as nn
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.typing import TensorType


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 initializer: Any = None,
                 activation_fn: Any = None,
                 use_bias: bool = True,
                 batch_norm: bool = False,
                 bias_init: float = 0.0):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size(int): Input size for FC Layer
            out_size (int): Output size for FC Layer
            initializer (Any): Initializer function for FC layer weights
            activation_fn (Any): Activation function at the end of layer
            use_bias (bool): Whether to add bias weights or not
            bias_init (float): Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        layers = []

        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)

        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)

        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)

        if batch_norm is True:
            layers.append(
                nn.BatchNorm1d(out_size),
            )

        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")

        if activation_fn is not None:
            layers.append(activation_fn())

        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)
