from niql.callbacks import NIQLCallbacks

MPE = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 64,
            'lr': 0.0005,
            'lr_schedule': [
                [0, 0.0005],
                [250000, 0.00005],
                [400000, 0.00001],
            ],
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
            'similarity_threshold': 0.999,
            'comm_dim': 0,
        }
    },
    'model_preference': {
        'core_arch': 'gru',  # mlp | gru
        "encode_layer": "64",  # for RNN model
        'hidden_state_size': 64,  # for RNN model
        'fcnet_activation': 'relu',
        'model': 'DRQNModel',
        # 'model': 'DuelingQFCN',
        'hidden_layer_dims': [64, 64],  # for mlp model
        'mixer_embedding': 256,  # for mixer model
        'mha_num_heads': 2,
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 500000,
    }
}