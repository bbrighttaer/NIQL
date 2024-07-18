from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from niql.models.obs_encoder import StraightThroughEstimator


class SimpleCommNet(nn.Module):

    def __init__(self, input_dim, comm_dim, agent_index, fp_size, discrete=False):
        super().__init__()
        comm_dim -= fp_size
        hdim = input_dim * 2
        self.model = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, comm_dim)
        )
        self.ste = StraightThroughEstimator() if discrete else lambda x: x
        self.fp = torch.eye(fp_size, fp_size).float()[agent_index].view(1, -1)

    def forward(self, input_x):
        x = self.model(input_x)
        x = self.ste(x)

        # add sender fingerprint
        fp = self.fp.to(x.device)
        fp = fp.repeat(*x.shape[:-1], 1)
        x = torch.cat([x, fp], dim=-1)

        return x


class ConcatCommMessagesAggregator(nn.Module):
    """Simply concatenates all messages"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, query, messages):
        """
        Aggregates neighbour message.

        :param query: query tensor of shape (bath_size, obs_dim)
        :param messages: local and shared neighbour messages, tensor of shape (bath_size, num_msgs, comm_dim)
        :return: aggregated neighbour messages, tensor of shape (batch_size, output_dim)
        """
        concat_msgs = messages.view(messages.shape[0], -1)
        return concat_msgs


class AttentionCommMessagesAggregator(nn.Module):
    """
    An attention-based neighbour messages aggregator.
    """

    def __init__(self, query_dim, comm_dim, hidden_dims: List[int], output_dim):
        super(AttentionCommMessagesAggregator, self).__init__()
        hidden_dim = hidden_dims[0]  # use the first val since multiple layers are not used here.
        self.fc_query = nn.Linear(query_dim, hidden_dim)
        self.fc_key = nn.Linear(comm_dim, hidden_dim)
        self.fc_value = nn.Linear(comm_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, query, messages):
        """
        Aggregates neighbour message.

        :param query: query tensor of shape (bath_size, obs_dim)
        :param messages: local and shared neighbour messages, tensor of shape (bath_size, num_msgs, comm_dim)
        :return: aggregated neighbour messages, tensor of shape (batch_size, output_dim)
        """
        # Transform the agent's observation into the query space
        Q = self.fc_query(query).unsqueeze(1)  # Shape: (batch_size, 1, hidden_dim)

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

        # ensure shape is (batch_size, output_dim)
        out = out.view(-1, out.shape[-1])

        return out


class GNNCommMessagesAggregator(nn.Module):
    """
    A GNN-based neighbour messages aggregator.
    """

    def __init__(self, query_dim: int, comm_dim: int, hidden_dims: List[int], output_dim: int):
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
        layers.extend([
            Aggregator(),
            nn.Linear(
                in_features=prev_layer_size,
                out_features=output_dim,
            )
        ])
        self.fcn = nn.Sequential(*layers)

    def forward(self, query, messages):
        """
        Aggregates neighbour message.

        :param query: query tensor of shape (bath_size, obs_dim)
        :param messages: local and shared neighbour messages, tensor of shape (bath_size, num_msgs, comm_dim)
        :return: aggregated neighbour messages, tensor of shape (batch_size, output_dim)
        """
        neighbour_msgs = messages[:, 1:, :].squeeze(1)
        h_ij = self.fcn(neighbour_msgs)
        return h_ij


class Aggregator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, msgs):
        x = torch.relu(torch.sum(msgs, dim=1))
        return x
