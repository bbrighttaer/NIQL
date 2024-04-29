from niql.callbacks import NIQLCallbacks

FINGERPRINT_SIZE = 2
MPE = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 128,
            'lr': 0.0005,
            'rollout_fragment_length': 1,
            'buffer_size': 10000,
            'target_network_update_freq': 200,
            'final_epsilon': 0.05,
            'epsilon_timesteps': 50000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': True,
            'gamma': 0.99,
            'lambda': 0.01,
            'tau': 0.5,
        }
    },
    'model_preference': {
        'core_arch': 'mlp',  # mlp | gru
        "encode_layer": "128-256",  # for RNN model
        'hidden_state_size': 256,  # for RNN model
        'fcnet_activation': 'relu',
        'model': 'FCN',
        'hidden_layer_dims': [256],  # for mlp model
        'mixer_embedding': 256,  # for mixer model
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 3000000,
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
            'final_epsilon': 1.0,
            'epsilon_timesteps': 1000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': False,
            'gamma': 0.99,
            'lambda': 0.5,
            'tau': 0.5,
            'beta': 0,
            'callbacks': NIQLCallbacks,
            'sharing_batch_size': 10,
        }
    },
    'model_preference': {
        'core_arch': 'mlp',  # mlp | gru
        'hidden_layer_dims': [64],  # for mlp model
        'mixer_embedding': 256,  # for mixer model
        'encode_layer': '32',  # for RNN model
        'hidden_state_size': 64,  # for RNN model
        'fcnet_activation': 'relu',
        'model': 'DuelingQFCN',
        # 'model': 'DRQNModel',
        # 'model': 'MatrixGameQMLP',
        # 'model': 'MatrixGameSplitQMLP'
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 10000,
    }
}
