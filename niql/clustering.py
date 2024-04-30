from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from ray.rllib import SampleBatch


def kmeans_cluster_batch(batch: SampleBatch, k) -> SampleBatch:
    """
    Applies K-Means clustering to the sample batch and optimistically reconcile rewards.

    :param batch: batch to be processed.
    :k: the number of clusters for the K-Means algorithm
    :return: updated batch.
    """
    # prep data
    batch = batch.copy()
    obs = batch[SampleBatch.OBS]
    actions = batch[SampleBatch.ACTIONS].reshape(-1, 1)
    rewards = batch[SampleBatch.REWARDS]
    data = np.concatenate([obs, actions], axis=1)

    # clustering
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans = kmeans.fit(data)
    labels = kmeans.predict(data)

    # optimistic reward mapping
    label_to_reward = defaultdict(int)
    for lbl, rew in zip(labels, rewards):
        if rew > label_to_reward[lbl]:
            label_to_reward[lbl] = rew
    opt_rewards = [label_to_reward[lbl] for lbl in labels]
    batch[SampleBatch.REWARDS] = np.array(opt_rewards)

    return batch
