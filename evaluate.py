"""
evaluate.py — Evaluate a trained model and print / save performance stats.

Usage:
    python evaluate.py --algo PPO
    python evaluate.py --algo DQN --episodes 20 --render
    python evaluate.py --algo SAC --model_path results/SAC/models/final_model
"""

import os
import argparse

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C, SAC, DDPG, TD3

from configs import CONFIGS
from reward_shaping_wrapper import reward_shaping_wrapper
from train import ALGORITHMS


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


def _default_model_base(
    algo_name: str,
    wrapper: bool,
    terminal_reward: bool,
    shaping_scale: float,
) -> str:
    run_variant = _run_variant_name(wrapper, terminal_reward, shaping_scale)
    if run_variant == "baseline":
        return os.path.join("results", algo_name)
    return os.path.join("results", algo_name, run_variant)


def evaluate(
    algo_name: str,
    model_path: str | None = None,
    n_episodes: int = 10,
    render: bool = False,
    wrapper: bool = False,
    terminal_reward: bool = False,
    shaping_scale: float = 0.05,

) -> dict[str, float]:
    """Load a trained model and evaluate it for *n_episodes* episodes.

    Args:
        algo_name:  Algorithm key (DQN, PPO, …).
        model_path: Path to a ``.zip`` model file (without extension).
                    Defaults to ``results/<algo>/models/best_model``.
        n_episodes: Number of evaluation episodes.
        render:     Whether to render the environment (requires a display).

    Returns:
        Dict with keys ``mean_reward``, ``std_reward``, ``mean_length``.
    """
    if algo_name not in CONFIGS:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. Choose from: {', '.join(CONFIGS)}"
        )
    if terminal_reward and not wrapper:
        raise ValueError("--terminal_reward requires --wrapper.")

    env_id = CONFIGS[algo_name]["env"]

    # ── resolve model path ───────────────────────────────────────────
    if model_path is None:
        base_dir = _default_model_base(
            algo_name,
            wrapper=wrapper,
            terminal_reward=terminal_reward,
            shaping_scale=shaping_scale,
        )
        best = os.path.join(base_dir, "models", "best_model")
        final = os.path.join(base_dir, "models", "final_model")
        model_path = best if os.path.exists(best + ".zip") else final
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'. "
            f"Run: python train.py --algo {algo_name}"
        )

    # ── load model ───────────────────────────────────────────────────
    AlgoClass = ALGORITHMS[algo_name]
    model = AlgoClass.load(model_path)
    print(f"Loaded {algo_name} from: {model_path}")

    # ── evaluation loop ──────────────────────────────────────────────
    render_mode = "human" if render else None
    
    env = gym.make(env_id, render_mode=render_mode)

    if wrapper:

        if terminal_reward:
            print("Applying reward shaping wrapper with terminal reward only.")
            shaping_scale = 0.0  # Disable dense shaping

        env = reward_shaping_wrapper(env, shaping_scale=shaping_scale)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            ep_length += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(
            f"  Episode {ep + 1:3d}/{n_episodes}  "
            f"reward={ep_reward:8.2f}  length={ep_length}"
        )

    env.close()

    stats = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    print(f"\n── {algo_name} evaluation results ({n_episodes} episodes) ──")
    print(f"  Mean reward : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean length : {stats['mean_length']:.1f} steps")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LunarLander agent."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(ALGORITHMS),
        help="RL algorithm to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model .zip (without extension). "
        "Defaults to the best model for the selected run variant.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--shaping_scale",
        type=float,
        default=0.05,
        help="Scale for reward shaping (default: 0.05).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (requires a display).",
    )
    parser.add_argument(
        "--wrapper",
        action="store_true",
        help="Apply reward shaping wrapper.",
    )
    parser.add_argument(
        "--terminal_reward",
        action="store_true",
        help="Use terminal reward only for reward shaping.",
    )
    args = parser.parse_args()
    evaluate(
        args.algo,
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render,
        wrapper=args.wrapper,
        terminal_reward=args.terminal_reward,
        shaping_scale=args.shaping_scale
    )
