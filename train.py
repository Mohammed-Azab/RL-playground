"""
train.py — Train a single RL algorithm on LunarLander.

Usage:
    python train.py --algo PPO
    python train.py --algo DQN --timesteps 300000 --seed 0
    python train.py --algo SAC  
"""

import os
import argparse

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from configs import CONFIGS
from reward_shaping_wrapper import reward_shaping_wrapper

ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3,
}


def _format_scale(scale: float) -> str:
    formatted = f"{scale:.3f}".rstrip("0").rstrip(".")
    return formatted.replace(".", "p")


def _run_variant_name(
    wrapper: bool,
    terminal_reward: bool,
    shaping_scale: float,
) -> str:
    if not wrapper:
        return "baseline"
    if terminal_reward:
        return "wrapped_terminal_reward"
    return f"wrapped_reward_shaping_s{_format_scale(shaping_scale)}"


def _result_dirs(algo_name: str, run_variant: str) -> tuple[str, str]:
    if run_variant == "baseline":
        base_dir = os.path.join("results", algo_name)
    else:
        base_dir = os.path.join("results", algo_name, run_variant)
    return os.path.join(base_dir, "logs"), os.path.join(base_dir, "models")


def train(
    algo_name: str,
    timesteps: int | None = None,
    seed: int = 42,
    wrapper: bool = False,
    terminal_reward: bool = False,
    shaping_scale: float = 0.05,
) -> None:
    """Train *algo_name* on the environment specified in CONFIGS and save the
    trained model plus episode logs to ``results/<algo_name>/``.

    Args:
        algo_name: One of DQN | PPO | A2C | SAC | DDPG | TD3.
        timesteps:  Override the default number of training timesteps.
        seed:       Random seed for reproducibility.
        wrapper:    Whether to apply the custom reward wrapper.
        terminal_reward: Use terminal reward only instead of dense shaping.
        shaping_scale: Scale applied to the dense shaping term.
    """
    if algo_name not in CONFIGS:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Choose from: {', '.join(CONFIGS)}"
        )
    if terminal_reward and not wrapper:
        raise ValueError("--terminal_reward requires --wrapper.")

    config = CONFIGS[algo_name]
    env_id = config["env"]
    hyperparams = config["hyperparams"].copy()
    policy = hyperparams.pop("policy")
    n_timesteps = timesteps if timesteps is not None else config["timesteps"]
    run_variant = _run_variant_name(wrapper, terminal_reward, shaping_scale)

    # ── directory setup ──────────────────────────────────────────────
    log_dir, model_dir = _result_dirs(algo_name, run_variant)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Algorithm : {algo_name}")
    print(f"  Env       : {env_id}")
    print(f"  Timesteps : {n_timesteps:,}")
    print(f"  Seed      : {seed}")
    print(f"  Variant   : {run_variant}")
    if wrapper:
        print(f"  Wrapper   : enabled")
        print(f"  TermOnly  : {terminal_reward}")
        print(f"  Scale     : {shaping_scale}")
    print(f"{'=' * 60}\n")

    # ── environments ─────────────────────────────────────────────────
    # Monitor wraps the env and records per-episode reward/length to CSV
    train_base_env = gym.make(env_id)
    eval_base_env = gym.make(env_id)

    if wrapper:
        effective_scale = 0.0 if terminal_reward else shaping_scale
        train_base_env = reward_shaping_wrapper(
            train_base_env,
            shaping_scale=effective_scale,
        )
        eval_base_env = reward_shaping_wrapper(
            eval_base_env,
            shaping_scale=effective_scale,
        )

    train_env = Monitor(train_base_env, log_dir)
    eval_env = Monitor(eval_base_env)

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
    parser.add_argument(
        "--wrapper",
        action="store_true",
        help="Apply the custom reward wrapper.",
    )
    parser.add_argument(
        "--terminal_reward",
        action="store_true",
        help="Use terminal reward only with the wrapper.",
    )
    parser.add_argument(
        "--shaping_scale",
        type=float,
        default=0.05,
        help="Scale for dense reward shaping (default: 0.05).",
    )
    args = parser.parse_args()
    train(
        args.algo,
        timesteps=args.timesteps,
        seed=args.seed,
        wrapper=args.wrapper,
        terminal_reward=args.terminal_reward,
        shaping_scale=args.shaping_scale,
    )
