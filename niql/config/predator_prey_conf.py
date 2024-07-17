from niql.callbacks import NIQLCallbacks

PREDATOR_PREY = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 32,
            'lr': 0.0005,
            'lr_schedule': [
                [0, 0.0005],
                [30000, 0.00005],
                [50000, 0.000005],
            ],
            'tdw_schedule': [
                [0, 1.0],
                [50000, 1.0],
                [200000, 0.0],
            ],
            'kde_subset_size': 100,
            'tdw_warm_steps': 3000,
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
            'similarity_threshold': 0.01,
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
        'tdw_vae': {
            'latent_dim': 16,
            'hdims': [128],
        },
        'mha_num_heads': 4,
        'add_action_dim': True,
        'use_vae_encoder': False,
        'comm_dim': 10,
        'comm_hdim': 64,
        'comm_aggregator': 'concat',
        'comm_aggregator_dim': 10,
        'comm_aggregator_hdims': [128],
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 1000000,
    }
}
