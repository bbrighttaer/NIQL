from niql.callbacks import NIQLCallbacks

SMAC = {
    "algo_parameters": {
        "algo_args": {
            "batch_episode": 32,
            "lr": 0.0005,
            "tdw_schedule": [
                [0, 1.0],
                [50000, 1.0],
                [100000, 0.0],
            ],
            "rollout_fragment_length": 1,
            "buffer_size": 5000,
            "target_network_update_freq": 1,
            "tau": 0.05,  # target network soft update
            "final_epsilon": 0.05,
            "epsilon_timesteps": 50000,
            "optimizer": "rmsprop",
            "reward_standardize": False,
            "gamma": 0.99,
            "lambda": 0.85,
            "callbacks": NIQLCallbacks,
        }
    },
    "model_preference": {
        "core_arch": "gru",  # mlp | gru
        "encode_layer": "64",  # for RNN model
        "hidden_state_size": 64,  # for RNN model
        "hidden_layer_dims": [128, 64],  # for mlp model
        "add_action_dim": False,
    },
    "stop_condition": {
        "episode_reward_mean": 2000,
        "timesteps_total": 2000000,
    }
}

