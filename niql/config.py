from niql.callbacks import NIQLCallbacks

FINGERPRINT_SIZE = 2
MPE = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 128,
            'lr': 0.0005,
            'rollout_fragment_length': 1,
            'buffer_size': 5000,
            'target_network_update_freq': 200,
            'final_epsilon': 0.05,
            'epsilon_timesteps': 50000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': True,
        }
    },
    'model_preference': {
        "core_arch": "gru",  # mlp | gru | lstm
        "encode_layer": "128-256",
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 3861200,
    }
}

COOP_MATRIX = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 128,
            'lr': 0.0005,
            'rollout_fragment_length': 1,
            'buffer_size': 1000,
            'target_network_update_freq': 100,
            'final_epsilon': 1.0,
            'epsilon_timesteps': 10000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': False,
            'gamma': 0.99,
            'callbacks': NIQLCallbacks,
        }
    },
    'model_preference': {
        'core_arch': 'mlp',
        'hidden_layer_dims': [64],
        # 'custom_model': 'MatrixGameQMLP',
        'custom_model': 'MatrixGameSplitQMLP'
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 10000,
    }
}
