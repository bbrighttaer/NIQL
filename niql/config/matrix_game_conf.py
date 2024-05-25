from niql.callbacks import NIQLCallbacks

MATRIX_GAME = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 64,
            'lr': 0.0005,
            'rollout_fragment_length': 1,
            'buffer_size': 1000,
            'target_network_update_freq': 10,
            'final_epsilon': 0.01,
            'epsilon_timesteps': 2000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': False,
            'gamma': 0.99,
            'lambda': 0.5,
            'tau': 0.5,
            'beta': 0,
            'callbacks': NIQLCallbacks,
            'sharing_batch_size': 10,
            'similarity_threshold': 0.999,
            'comm_dim': 0,
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