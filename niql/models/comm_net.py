from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from requests import get

from niql.models.obs_encoder import StraightThroughEstimator


class SimpleCommNet(nn.Module):

    def __init__(self, input_dim, hdim, com_dim, discrete=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, com_dim)
        self.ste = StraightThroughEstimator() if discrete else lambda x: x

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.ste(self.fc3(x))
        return x


class AttentionCommMessagesAggregator(nn.Module):
    """
    An attention-based neighbour messages aggregator.
    """

    def __init__(self, obs_dim, comm_dim, hidden_dim, output_dim):
        super(AttentionCommMessagesAggregator, self).__init__()
        self.fc_query = nn.Linear(obs_dim, hidden_dim)
        self.fc_key = nn.Linear(comm_dim, hidden_dim)
        self.fc_value = nn.Linear(comm_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs, messages):
        """
        Aggregates neighbour message.

        :param obs: Local agent observation, tensor of shape (bath_size, obs_dim)
        :param messages: local and shared neighbour messages, tensor of shape (bath_size, num_msgs, comm_dim)
        :return: aggregated neighbour messages, tensor of shape (batch_size, output_dim)
        """
        # Transform the agent's observation into the query space
        Q = self.fc_query(obs).unsqueeze(1)  # Shape: (batch_size, hidden_dim)

        # Transform the neighbor observations into the key and value spaces
        K = self.fc_key(messages)
        V = self.fc_value(messages)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.permute(0, 2, 1))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum of values
        context = torch.matmul(attention_weights, V)

        # Pass the context through the output layer
        out = F.relu(self.fc_out(context))

        return out


class GNNCommMessagesAggregator(nn.Module):
    """
    A GNN-based neighbour messages aggregator.
    """

    def __init__(self, obs_dim: int, comm_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []

        # hidden layers
        prev_layer_size = comm_dim
        for size in hidden_dims:
            layers.extend([
                nn.Linear(
                    in_features=prev_layer_size,
                    out_features=size,
                ),
                nn.ReLU(),
            ])
            prev_layer_size = size

        # output layer
        layers.append(nn.Linear(
            in_features=prev_layer_size,
            out_features=output_dim,
        ))
        self.fcn = nn.Sequential(*layers)

    def forward(self, obs, messages):
        """
        Aggregates neighbour message.

        :param obs: Local agent observation, tensor of shape (bath_size, obs_dim)
        :param messages: local and shared neighbour messages, tensor of shape (bath_size, num_msgs, comm_dim)
        :return: aggregated neighbour messages, tensor of shape (batch_size, output_dim)
        """
        v_i = messages[:, 0, :]
        neighbour_msgs = messages[:, 1:, :]

        h_ij = self.fcn(neighbour_msgs)
        h_ij = torch.relu(torch.sum(h_ij, dim=1))
        # v_i = self.fcn(v_i)
        v_i = torch.relu(v_i + h_ij)

        return v_i
