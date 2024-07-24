from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder

from niql.models.base_torch_model import BaseTorchModel

torch, nn = try_import_torch()


class JointQRNNComm(BaseTorchModel):
    """The default GRU model for Joint Q."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]

        # only support gru cell
        model_arch_args = self.custom_config["model_arch_args"]
        if model_arch_args["core_arch"] != "gru":
            raise ValueError(
                "core arch should be gru, got {}".format(model_arch_args["core_arch"]))

        self.activation = model_config.get("fcnet_activation")
        self.hidden_state_size = model_arch_args["hidden_state_size"]
        comm_dim = model_arch_args["comm_dim"]

        # lookup tables
        self.obs_lookup = nn.Embedding(2, self.hidden_state_size)
        self.prev_action_lookup = nn.Embedding(num_outputs, self.hidden_state_size)
        self.prev_message_lookup = nn.Embedding(comm_dim, self.hidden_state_size)
        self.agent_lookup = nn.Embedding(self.n_agents, self.hidden_state_size)

        # mlp for messages
        layers = []
        aggregated_msgs_dim = comm_dim * (self.n_agents - 1)
        if model_arch_args.get("use_msgs_bn", True):
            layers.append(
                nn.BatchNorm1d(aggregated_msgs_dim)
            )
        layers.append(
            SlimFC(
                in_size=aggregated_msgs_dim,
                out_size=self.hidden_state_size,
                initializer=normc_initializer(0.01),
                activation_fn=self.activation,
            )
        )
        self.messages_mlp = nn.Sequential(*layers)

        # set up RNN
        dropout_rate = self.custom_config.get("dropout_rate") or 0
        self.rnn = nn.GRU(input_size=self.hidden_state_size, hidden_size=self.hidden_state_size,
                          num_layers=2, dropout=dropout_rate, batch_first=True)

        # set up outputs
        self.outputs = nn.Sequential()
        if dropout_rate > 0:
            self.outputs.add_module('dropout1', nn.Dropout(dropout_rate))
        self.outputs.add_module(
            'linear1',
            SlimFC(
                in_size=self.hidden_state_size,
                out_size=self.hidden_state_size,
                initializer=normc_initializer(0.01),
                activation_fn=self.activation,
            )
        )
        if model_arch_args.get("use_msgs_bn", True):
            self.outputs.add_module('batchnorm1', nn.BatchNorm1d(self.hidden_state_size))
        self.outputs.add_module('relu1', nn.ReLU(inplace=True))
        self.outputs.add_module(
            'linear2',
            SlimFC(
                in_size=self.hidden_state_size,
                out_size=num_outputs + comm_dim,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            )
        )

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h0 = self.rnn.all_weights[0][0].new(
            self.n_agents, self.hidden_state_size
        ).zero_().squeeze(0)
        hidden_state = [h0, h0.detach().clone()]
        return hidden_state

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens, agents_idx, prev_action_batch, prev_msgs, messages):
        inputs = input_dict["obs_flat"].float()
        if len(self.full_obs_space.shape) == 3:  # 3D
            inputs = inputs.reshape((-1,) + self.full_obs_space.shape)

        # obs lookup
        obs_emb = self.obs_lookup(inputs.long()).view(-1, self.hidden_state_size)

        # hidden state
        h = torch.stack(hidden_state, dim=0)

        # prev action lookup
        act_emb = self.prev_action_lookup(prev_action_batch.view(-1,).long())

        # agents lookup
        agents_emb = self.agent_lookup(agents_idx.reshape(-1,))

        # prev message lookup
        prev_msgs_emb = self.prev_message_lookup(prev_msgs.argmax(dim=1))

        # received messages
        msgs_emb = self.messages_mlp(messages)

        # add embeddings
        z = agents_emb + obs_emb + act_emb + prev_msgs_emb + msgs_emb

        # rnn phase
        z_out, h = self.rnn(z.unsqueeze(1), h)
        h = torch.split(h, [1, 1])

        # mlp output
        q = self.outputs(z_out.squeeze(1))

        return q, [*h]
