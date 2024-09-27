
STOCHASTIC_GAME = {
    "algo_parameters": {
        "algo_args": {
            "batch_episode": 32,
            "lr": 0.0005,
            "tdw_schedule": [
                [0, 1.0],
                [50000, 1.0],
                [60000, 0.1],
            ],
            "tdw_eps": 0.1,
            "rollout_fragment_length": 1,
            "buffer_size": 5000,
            "soft_target_update": False,
            "target_network_update_freq": 200,
            "tau": 0.01,  # target network soft update
            "final_epsilon": 0.05,
            "epsilon_timesteps": 50000,
            "optimizer": "rmsprop",
            "reward_standardize": False,
            "gamma": 0.99,
            "lambda": 0.5,
            "add_action_dim": True,
        }
    },
    "model_preference": {
        "core_arch": "gru",  # mlp | gru
        "encode_layer": "64",  # for RNN model
        "hidden_state_size": 64,  # for RNN model
        "hidden_layer_dims": [128, 64],  # for mlp model
    },
    "stop_condition": {
        "episode_reward_mean": 2000,
        "timesteps_total": 1050000,
    }
}

