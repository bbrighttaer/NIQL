mpe = {
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
        "core_arch": "mlp",
        "encode_layer": "128-256",
    },
    'stop_condition': {
        'episode_reward_mean': 2000,
        'timesteps_total': 3861200,
    }
}
