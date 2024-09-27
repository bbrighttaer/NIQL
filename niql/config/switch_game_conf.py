
SWITCH_RIDDLE = {
    'algo_parameters': {
        'algo_args': {
            'batch_episode': 32,
            'lr': 0.0005,
            'rollout_fragment_length': 1,
            'buffer_size': 5000,
            'target_network_update_freq': 100,
            'final_epsilon': 0.005,
            'epsilon_timesteps': 50000,
            'optimizer': 'rmsprop',  # "rmsprop | adam"
            'reward_standardize': True,
            'gamma': 1,
            'simple_optimizer': True,
        }
    },
    'model_preference': {
        'core_arch': 'gru',
        "encode_layer": "128",
        'hidden_state_size': 128,
        'fcnet_activation': 'relu',
        'model': 'DRQNModel',
        'hidden_layer_dims': [64, 64],
        'comm_dim': 2,
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 1000000,
    }
}
