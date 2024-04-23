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
            'buffer_size': 10000,
            'target_network_update_freq': 10,
            'final_epsilon': 0.01,
            'epsilon_timesteps': 1000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': False,
            'gamma': 0.99,
            'lambda': 0.01,
            'tau': 0.5,
            'callbacks': NIQLCallbacks,
        }
    },
    'model_preference': {
        'core_arch': 'gru',  # mlp | gru | lstm
        'hidden_layer_dims': [64],  # for mlp model
        'mixer_embedding': 256,  # for mixer model
        'encode_layer': '128',  # for RNN model
        'hidden_state_size': 256,  # for RNN model
        'fcnet_activation': 'relu',
        'custom_model': 'DRQNModel',
        # 'custom_model': 'MatrixGameQMLP',
        # 'custom_model': 'MatrixGameSplitQMLP'
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 10000,
    }
}
