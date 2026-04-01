"""
train.py — Train a single RL algorithm on LunarLander.

Usage:
    python train.py --algo PPO
    python train.py --algo DQN --timesteps 300000 --seed 0
    python train.py --algo SAC  # uses LunarLanderContinuous-v2 automatically
"""

import os
import argparse

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from configs import CONFIGS

ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3,
}


def train(algo_name: str, timesteps: int | None = None, seed: int = 42) -> None:
    """Train *algo_name* on the environment specified in CONFIGS and save the
    trained model plus episode logs to ``results/<algo_name>/``.

    Args:
        algo_name: One of DQN | PPO | A2C | SAC | DDPG | TD3.
        timesteps:  Override the default number of training timesteps.
        seed:       Random seed for reproducibility.
    """
    if algo_name not in CONFIGS:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Choose from: {', '.join(CONFIGS)}"
        )

    config = CONFIGS[algo_name]
    env_id = config["env"]
    hyperparams = config["hyperparams"].copy()
    policy = hyperparams.pop("policy")
    n_timesteps = timesteps if timesteps is not None else config["timesteps"]

    # ── directory setup ──────────────────────────────────────────────
    log_dir = os.path.join("results", algo_name, "logs")
    model_dir = os.path.join("results", algo_name, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Algorithm : {algo_name}")
    print(f"  Env       : {env_id}")
    print(f"  Timesteps : {n_timesteps:,}")
    print(f"  Seed      : {seed}")
    print(f"{'=' * 60}\n")

    # ── environments ─────────────────────────────────────────────────
    # Monitor wraps the env and records per-episode reward/length to CSV
    train_env = Monitor(gym.make(env_id), log_dir)
    eval_env = Monitor(gym.make(env_id))

    # ── model ────────────────────────────────────────────────────────
    AlgoClass = ALGORITHMS[algo_name]
    model = AlgoClass(policy, train_env, seed=seed, verbose=1, **hyperparams)

    # ── callbacks ────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix=algo_name.lower(),
    )

    # ── training ─────────────────────────────────────────────────────
    model.learn(
        total_timesteps=n_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    # ── save final model ─────────────────────────────────────────────
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\n✓ {algo_name} training complete.")
    print(f"  Model  → {final_path}.zip")
    print(f"  Logs   → {log_dir}/\n")

    train_env.close()
    eval_env.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a single RL algorithm on LunarLander."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(ALGORITHMS),
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Number of training timesteps (overrides config default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()
    train(args.algo, timesteps=args.timesteps, seed=args.seed)
