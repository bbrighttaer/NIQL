from typing import List

from ray.rllib.execution import ReplayBuffer
from ray.rllib.utils.typing import SampleBatchType


class CERTReplayBuffer(ReplayBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_sampled = 0

    def sample(self, idxes: List[int]) -> SampleBatchType:
        """Sample a batch of experiences.

        Args:
            idxes: the IDs of the selected samples

        Returns:
            SampleBatchType: concatenated batch of items.
        """
        self._num_sampled += len(idxes)
        return self._encode_sample(idxes)
