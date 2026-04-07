"""
Hyperparameter configurations for each RL algorithm on LunarLander.

DQN is fundamentally for discrete actions.
PPO and A2C can work in both settings in theory.
SAC, DDPG, and TD3 are continuous-control algorithms.

Discrete-action algorithms (LunarLander-v3):
  DQN, PPO, A2C

Continuous-action algorithms (LunarLanderContinuous-v3):
  SAC, DDPG, TD3

  using MlpPolicy for all algorithms, which is Multi-Layer Perceptron Policy (a feedforward neural network).
"""

CONFIGS = {
    # ------------------------------------------------------------------
    # Deep Q-Network  (discrete actions)
    # ------------------------------------------------------------------
    "DQN": {
        "env": "LunarLander-v3",
        "timesteps": 500_000,
        "hyperparams": {
            "policy": "MlpPolicy",
            "learning_rate": 6.3e-4,
            "buffer_size": 50_000,
            "learning_starts": 0,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": -1,
            "exploration_fraction": 0.12,
            "exploration_final_eps": 0.1,
        },
    },
    # ------------------------------------------------------------------
    # Proximal Policy Optimization  (discrete actions)
    # ------------------------------------------------------------------
    "PPO": {
        "env": "LunarLanderContinuous-v3",
        "timesteps": 1_000_000,
        "hyperparams": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.999,
            "gae_lambda": 0.98,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
    # ------------------------------------------------------------------
    # Advantage Actor-Critic  (discrete actions)
    # ------------------------------------------------------------------
    "A2C": {
        "env": "LunarLander-v3",
        "timesteps": 1_000_000,
        "hyperparams": {
            "policy": "MlpPolicy",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.0,
            "normalize_advantage": False,
        },
    },
    # ------------------------------------------------------------------
    # Soft Actor-Critic  (continuous actions)
    # ------------------------------------------------------------------
    "SAC": {
        "env": "LunarLanderContinuous-v3",
        "timesteps": 500_000,
        "hyperparams": {
            "policy": "MlpPolicy",
            "learning_rate": 1e-3,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    },
    # ------------------------------------------------------------------
    # Deep Deterministic Policy Gradient  (continuous actions)
    # ------------------------------------------------------------------
    "DDPG": {
        "env": "LunarLanderContinuous-v3",
        "timesteps": 500_000,
        "hyperparams": {
            "policy": "MlpPolicy",
            "learning_rate": 1e-3,
            "buffer_size": 200_000,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.98,
            "train_freq": (1, "episode"),
            "gradient_steps": -1,
        },
    },
    # ------------------------------------------------------------------
    # Twin Delayed Deep Deterministic Policy Gradient  (continuous actions)
    # ------------------------------------------------------------------
    "TD3": {
        "env": "LunarLanderContinuous-v3",
        "timesteps": 500_000,
        "hyperparams": {
            "policy": "MlpPolicy",
            "learning_rate": 1e-3,
            "buffer_size": 200_000,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.98,
            "train_freq": (1, "episode"),
            "gradient_steps": -1,
            "policy_delay": 2,
        },
    },
}
