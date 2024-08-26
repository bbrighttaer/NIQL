from niql.callbacks import NIQLCallbacks

default_config = {
    "algo_parameters": {
        "algo_args": {
            "batch_episode": 32,
            "lr": 0.0001,
            "lr_schedule": [
                [0, 0.001],
                [200000, 0.00001]
            ],
            "tdw_schedule": [
                [0, 1.0],
                [50000, 1.0],
                [60000, 0.0],
            ],
            "tdw_bandwidth": 5.,
            "rollout_fragment_length": 1,
            "buffer_size": 50000,
            "enable_stochastic_eviction": True,
            "target_network_update_freq": 10,
            "final_epsilon": 0.1,
            "epsilon_timesteps": 50000,
            "optimizer": "adam",
            "reward_standardize": False,
            "gamma": 0.99,
            "lambda": 0.6,
            "tau": 0.5,  # target network soft update
            "beta": 0,
            "callbacks": NIQLCallbacks,
            "sharing_batch_size": 10,
            "similarity_threshold": 0.999,
        }
    },
    "model_preference": {
        "core_arch": "mlp",  # mlp | gru
        "encode_layer": "128",  # for RNN model
        "hidden_state_size": 64,  # for RNN model
        "fcnet_activation": "relu",
        "hidden_layer_dims": [128, 64],  # for mlp model
        "mixer_embedding": 256,  # for mixer model
        "add_action_dim": False,
        "comm_dim": 0,
        "comm_hdim": 64,
        "comm_aggregator_dim": 10,
        "comm_aggregator_hdims": [128],
    },
    "stop_condition": {
        "episode_reward_mean": 2000,
        "timesteps_total": 1000000,
    }
}

REGISTRY = {}
