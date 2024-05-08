from niql.callbacks import NIQLCallbacks

FINGERPRINT_SIZE = 2
MPE = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 32,
            'lr': 0.0005,
            'rollout_fragment_length': 1,
            'buffer_size': 5000,
            'target_network_update_freq': 200,
            'final_epsilon': 0.05,
            'epsilon_timesteps': 50000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': True,
            'gamma': 0.99,
            'lambda': 0.6,
            'tau': 0.5,  # target network soft update
            'beta': 0,
            'callbacks': NIQLCallbacks,
            'sharing_batch_size': 10,
            'similarity_threshold': 0.9999,
        }
    },
    'model_preference': {
        'core_arch': 'gru',  # mlp | gru
        "encode_layer": "64",  # for RNN model
        'hidden_state_size': 64,  # for RNN model
        'fcnet_activation': 'relu',
        'model': 'DRQNModel',
        # 'model': 'DuelingQFCN',
        'hidden_layer_dims': [64],  # for mlp model
        'mixer_embedding': 256,  # for mixer model
        'mha_num_heads': 4,
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 300000,
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
            'lambda': 0.6,
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
        'encode_layer': '64',  # for RNN model
        'hidden_state_size': 64,  # for RNN model
        'fcnet_activation': 'relu',
        'model': 'DuelingQFCN',
        # 'model': 'DRQNModel',
        # 'model': 'MatrixGameQMLP',
        # 'model': 'MatrixGameSplitQMLP',
        'mha_num_heads': 2,
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 10000,
    }
}
