from niql.callbacks import NIQLCallbacks

MATRIX_GAME = {
    "algo_parameters": {
        "algo_args": {
            "batch_episode": 32,
            "lr": 0.0005,
            "tdw_schedule": [
                [0, 1.0],
                [1000, 1.0],
                [3000, 0.],
            ],
            "tdw_eps": 0.1,
            "rollout_fragment_length": 1,
            "buffer_size": 10000,
            "target_network_update_freq": 1,
            "tau": 0.01,  # target network soft update
            "final_epsilon": 0.01,
            "epsilon_timesteps": 10000,
            "optimizer": "rmsprop",
            "reward_standardize": False,
            "gamma": 0.99,
            "lambda": 0.65,
            "callbacks": NIQLCallbacks,
            "sharing_batch_size": 10,
            "similarity_threshold": 0.999,
        }
    },
    "model_preference": {
        "core_arch": "mlp",  # mlp | gru
        "encode_layer": "64",  # for RNN model
        "hidden_state_size": 64,  # for RNN model
        "hidden_layer_dims": [64],  # for mlp model
        "add_action_dim": False,
    },
    "stop_condition": {
        "episode_reward_mean": 2000,
        "timesteps_total": 100000,
    }
}
