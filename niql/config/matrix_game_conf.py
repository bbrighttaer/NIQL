from niql.callbacks import NIQLCallbacks

MATRIX_GAME = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 32,
            'lr': 0.0005,
            # 'lr_schedule': [
            #     [0, 0.0005],
            #     [1000, 0.00005],
            # ],
            'tdw_schedule': [
                [0, 1.0],
                [2000, 1.0],
                [3000, 0.0],
            ],
            'kde_subset_size': 10,
            'tdw_warm_steps': 1000,
            'rollout_fragment_length': 1,
            'buffer_size': 5000,
            'enable_stochastic_eviction': False,
            'target_network_update_freq': 10,
            'final_epsilon': 0.01,
            'epsilon_timesteps': 3000,
            'tdw_timesteps': 5000,
            'optimizer': 'rmsprop',  # "adam"
            'reward_standardize': False,
            'gamma': 1.0,
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
        # 'model': 'DuelingQFCN',
        # 'model': 'DRQNModel',
        'model': 'MatrixGameQMLP',
        # 'model': 'MatrixGameSplitQMLP',
        'tdw_vae': {
            'latent_dim': 2,
            'hdims': [16],
        },
        'mha_num_heads': 2,
        'add_action_dim': False,
        'comm_dim': 0,
        'comm_hdim': 64,
        'comm_aggregator_dim': 10,
        'comm_aggregator_hdims': [128, 32],
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 10000,
    }
}