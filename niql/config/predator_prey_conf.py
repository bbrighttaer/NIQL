from niql.callbacks import NIQLCallbacks

PREDATOR_PREY = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 32,
            'lr': 0.0005,
            # 'lr_schedule': [
            #     [0, 0.005],
            #     # [250000, 0.00005],
            #     [500000, 0.0005],
            # ],
            'rollout_fragment_length': 1,
            'buffer_size': 5000,
            'enable_stochastic_eviction': False,
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
            'comm_dim': 10,
            'comm_aggregator_dim': 16,
            'lds_timesteps': 50000
        }
    },
    'model_preference': {
        'core_arch': 'gru',  # mlp | gru
        "encode_layer": "128",  # for RNN model
        'hidden_state_size': 128,  # for RNN model
        'fcnet_activation': 'relu',
        'model': 'DRQNModel',
        # 'model': 'DuelingQFCN',
        'hidden_layer_dims': [64],  # for mlp model
        'mixer_embedding': 256,  # for mixer model
        'mha_num_heads': 4,
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 500000,
    }
}
