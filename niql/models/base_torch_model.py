from typing import Union, List, Any
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import restore_original_dimensions, flatten
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelInputDict, TensorType


class BaseTorchModel(TorchModelV2, nn.Module):

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
