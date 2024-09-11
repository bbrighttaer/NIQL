# MIT License

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

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class JointQRNN(TorchModelV2, nn.Module):
    """The default GRU model for Joint Q."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = model_config["n_agents"]

        # only support gru cell
        if self.custom_config["model_arch_args"]["core_arch"] != "gru":
            raise ValueError(
                "core arch should be gru, got {}".format(self.custom_config["model_arch_args"]["core_arch"]))

        # encoder
        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]
        self.encoder = nn.Linear(self.full_obs_space.shape[0], self.hidden_state_size)
        self.rnn = nn.GRUCell(self.hidden_state_size, self.hidden_state_size)
        self.q_value = nn.Linear(self.hidden_state_size, num_outputs)

    def get_initial_state(self):
        # Place hidden states on same device as model.
        hidden_state = [
            self.q_value.weight.new(self.n_agents,
                                    self.hidden_state_size).zero_()
        ]
        return hidden_state

    def forward(self, input_dict, hidden_state, seq_lens):
        inputs = input_dict["obs_flat"].float()
        if len(self.full_obs_space.shape) == 3:  # 3D
            inputs = inputs.reshape((-1,) + self.full_obs_space.shape)
        h_in = hidden_state[0].reshape(-1, self.hidden_state_size)
        x = torch.relu(self.encoder(inputs))
        h = self.rnn(x, h_in)
        q = self.q_value(h)
        return q, [h]
