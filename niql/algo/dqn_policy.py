from typing import Dict, Optional, Tuple

import torch
from marllib.patch.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.policy import Policy, TorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import override, DeveloperAPI
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import TensorType, ModelGradients, AgentID


class IQLPolicy(TorchPolicy):

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:

        # Set Model to train mode.
        if self.model:
            self.model.train()

        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self,
            train_batch=postprocessed_batch,
            result=learn_stats,
        )

        # Compute gradients (will calculate all losses and `backward()`
        # them to get the grads).
        grads, fetches = self.compute_gradients(postprocessed_batch)

        # Step the optimizers.
        self.apply_gradients(_directStepOptimizerSingleton)

        if self.model:
            fetches["model"] = self.model.metrics()
        fetches.update({"custom_metrics": learn_stats})

        return fetches

    def apply_gradients(self, gradients: ModelGradients) -> None:
        if gradients == _directStepOptimizerSingleton:
            for i, opt in enumerate(self._optimizers):
                opt.step()
        else:
            # TODO(sven): Not supported for multiple optimizers yet.
            assert len(self._optimizers) == 1
            for g, p in zip(gradients, self.model.parameters()):
                if g is not None:
                    if torch.is_tensor(g):
                        p.grad = g.to(self.device)
                    else:
                        p.grad = torch.from_numpy(g).to(self.device)

            self._optimizers[0].step()

    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[
                "Policy", SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        return sample_batch
