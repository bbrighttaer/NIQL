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
"""
From marllib
"""
from typing import Union, List, Any

from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions, flatten
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelInputDict
from torch import TensorType

from niql.models.base_encoder import BaseEncoder
from niql.models.fds import FDS

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
        self.n_agents = self.custom_config["num_agents"]

        # only support gru cell
        if self.custom_config["model_arch_args"]["core_arch"] != "gru":
            raise ValueError(
                "core arch should be gru, got {}".format(self.custom_config["model_arch_args"]["core_arch"]))

        self.activation = model_config.get("fcnet_activation")

        # encoder
        self.encoder = BaseEncoder(model_config, {'obs': self.full_obs_space})
        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]
        self.rnn = nn.GRUCell(self.encoder.output_dim, self.hidden_state_size)

        # Feature Distribution Smoothing
        fds_config = model_config.get("fds", None)
        if fds_config:
            self.FDS = FDS(feature_dim=self.hidden_state_size, **fds_config)
        else:
            self.FDS = None

        self.q_value = SlimFC(
            in_size=self.hidden_state_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        # record the custom config
        if self.custom_config["global_state_flag"]:
            state_dim = self.custom_config["space_obs"]["state"].shape
        else:
            state_dim = self.custom_config["space_obs"]["obs"].shape
        self.raw_state_dim = state_dim

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        hidden_state = [
            self.q_value._model._modules["0"].weight.new(self.n_agents,
                                                         self.hidden_state_size).zero_().squeeze(0)
        ]
        return hidden_state

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens, labels=None, epoch=None):
        inputs = input_dict["obs_flat"].float()
        if len(self.full_obs_space.shape) == 3:  # 3D
            inputs = inputs.reshape((-1,) + self.full_obs_space.shape)
        x = self.encoder(inputs)
        h_in = hidden_state[0].reshape(-1, self.hidden_state_size)
        h = self.rnn(x, h_in)
        h_smoothed = h
        if self.training and self.FDS is not None:
            h_smoothed = self.FDS.smooth(h, labels, epoch)
        q = self.q_value(h_smoothed)
        return q, [h]

    def __call__(
            self,
            input_dict: Union[SampleBatch, ModelInputDict],
            state: List[Any] = None,
            seq_lens: TensorType = None,
            **kwargs,
    ) -> (TensorType, List[TensorType]):
        """Call the model with the given input tensors and state.

                This is the method used by RLlib to execute the forward pass. It calls
                forward() internally after unpacking nested observation tensors.

                Custom models should override forward() instead of __call__.

                Args:
                    input_dict (Union[SampleBatch, ModelInputDict]): Dictionary of
                        input tensors.
                    state (list): list of state tensors with sizes matching those
                        returned by get_initial_state + the batch dimension
                    seq_lens (Tensor): 1D tensor holding input sequence lengths.

                Returns:
                    (outputs, state): The model output tensor of size
                        [BATCH, output_spec.size] or a list of tensors corresponding to
                        output_spec.shape_list, and a list of state tensors of
                        [BATCH, state_size_i].
                """

        # Original observations will be stored in "obs".
        # Flattened (preprocessed) obs will be stored in "obs_flat".

        # SampleBatch case: Models can now be called directly with a
        # SampleBatch (which also includes tracking-dict case (deprecated now),
        # where tensors get automatically converted).
        if isinstance(input_dict, SampleBatch):
            restored = input_dict.copy(shallow=True)
            # Backward compatibility.
            if seq_lens is None:
                seq_lens = input_dict.get(SampleBatch.SEQ_LENS)
            if not state:
                state = []
                i = 0
                while "state_in_{}".format(i) in input_dict:
                    state.append(input_dict["state_in_{}".format(i)])
                    i += 1
            input_dict["is_training"] = input_dict.is_training
        else:
            restored = input_dict.copy()

        # No Preprocessor used: `config._disable_preprocessor_api`=True.
        # TODO: This is unnecessary for when no preprocessor is used.
        #  Obs are not flat then anymore. However, we'll keep this
        #  here for backward-compatibility until Preprocessors have
        #  been fully deprecated.
        if self.model_config.get("_disable_preprocessor_api"):
            restored["obs_flat"] = input_dict["obs"]
        # Input to this Model went through a Preprocessor.
        # Generate extra keys: "obs_flat" (vs "obs", which will hold the
        # original obs).
        else:
            restored["obs"] = restore_original_dimensions(
                input_dict["obs"], self.obs_space, self.framework)
            try:
                if len(input_dict["obs"].shape) > 2:
                    restored["obs_flat"] = flatten(input_dict["obs"], self.framework)
                else:
                    restored["obs_flat"] = input_dict["obs"]
            except AttributeError:
                restored["obs_flat"] = input_dict["obs"]

        with self.context():
            res = self.forward(restored, state or [], seq_lens, **kwargs)

        if ((not isinstance(res, list) and not isinstance(res, tuple))
                or len(res) != 2):
            raise ValueError(
                "forward() must return a tuple of (output, state) tensors, "
                "got {}".format(res))
        outputs, state_out = res

        if not isinstance(state_out, list):
            raise ValueError(
                "State output is not a list: {}".format(state_out))

        self._last_output = outputs
        return outputs, state_out if len(state_out) > 0 else (state or [])
