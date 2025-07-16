def envs():
    return {
        "FrozenLake": {
            "env": {
                "env_name": "FrozenLake-v1",
                "env_config": {
                    "size": 14,
                    "p": 0.87,
                },
                "seed": 42,
            },
            "run": {
                "max_episodes": 1_000_000,  # maximum training episode
                "max_steps": 1_000_000,  # maximum training step
                "max_steps_per_episode": 100_000,  # maximum steps per episode
                "max_time": 60 * 60 * 4,  # maximum training time in seconds
            },
        },
        "CartPole": {
            "env": {
                "env_name": "CartPole-v1",
                "env_config": {},
                "seed": 43,
            },
            "run": {
                "max_episodes": 1_000,  # maximum training episode
                "max_steps": 50_000,  # maximum training step
                "max_steps_per_episode": 100_000,  # maximum steps per episode
                "max_time": 60 * 60 * 0.2,  # maximum training time in seconds
            },
        },
        "MountainCar": {
            "env": {
                "env_name": "MountainCar-v0",
                "env_config": {},
                "seed": 43,
            },
            "run": {
                "max_episodes": 100_000,  # maximum training episode
                "max_steps": 300_000,  # maximum training step
                "max_steps_per_episode": 100_000,  # maximum steps per episode
                "max_time": 60 * 60 * 2.5,  # maximum training time in seconds
            },
        },
        "SeaQuest": {
            "env": {
                "env_name": "Seaquest-ramNoFrameskip-v4",
                "env_config": {},
                "seed": 43,
            },
            "run": {
                "max_episodes": 1_000_000,  # maximum training episode
                "max_steps": 2_000_000,  # maximum training step
                "max_steps_per_episode": 1_000_000,  # maximum steps per episode
                "max_time": 60 * 60 * 8,  # maximum training time in seconds
            },
        },
        "LunarLander": {
            "env": {
                "env_name": "LunarLander-v3",
                "env_config": {},
                "seed": 43,
            },
            "run": {
                "max_episodes": 100_000,  # maximum training episode
                "max_steps": 200_000,  # maximum training step
                "max_steps_per_episode": 100_000,  # maximum steps per episode
                "max_time": 60 * 60 * 2,  # maximum training time in seconds
            },
        },
    }