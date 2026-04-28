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
from icm import icm_wrapper
from train import ALGORITHMS


def _run_variant_name(
    wrapper: bool,
    terminal_reward: bool,
    icm: bool = False,
    icm_only: bool = False,
) -> str:
    if icm:
        if wrapper and terminal_reward:
            return "wrapped_icm_only_terminal" if icm_only else "wrapped_icm_terminal"
        return "wrapped_icm_only" if icm_only else "wrapped_icm"
    if not wrapper:
        return "baseline"
    if terminal_reward:
        return "wrapped_terminal_reward"
    return "wrapped_reward_shaping"


def _default_model_base(
    algo_name: str,
    wrapper: bool,
    terminal_reward: bool,
    icm: bool = False,
    icm_only: bool = False,
    seed: int = 42,
) -> str:
    run_variant = _run_variant_name(wrapper, terminal_reward, icm, icm_only)
    folder = f"{run_variant}_s{seed}"
    return os.path.join("results", algo_name, folder)


def evaluate(
    algo_name: str,
    model_path: str | None = None,
    n_episodes: int = 10,
    render: bool = False,
    wrapper: bool = False,
    terminal_reward: bool = False,
    icm: bool = False,
    icm_only: bool = False,
    icm_beta: float = 1.0,
    seed: int = 42,
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

    # Resolve model path
    if model_path is None:
        base_dir = _default_model_base(
            algo_name,
            wrapper=wrapper,
            terminal_reward=terminal_reward,
            icm=icm,
            icm_only=icm_only,
            seed=seed,
        )
        best = os.path.join(base_dir, "models", "best_model")
        final = os.path.join(base_dir, "models", "final_model")
        model_path = best if os.path.exists(best + ".zip") else final
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'. "
            f"Run: python train.py --algo {algo_name}"
        )

    # Load model
    AlgoClass = ALGORITHMS[algo_name]
    model = AlgoClass.load(model_path)
    print(f"Loaded {algo_name} from: {model_path}")

    # Evaluation loop
    render_mode = "human" if render else None
    
    env = gym.make(env_id, render_mode=render_mode)

    if icm:
        if wrapper and terminal_reward:
            env = reward_shaping_wrapper(env)
        icm_cfg = CONFIGS[algo_name].get("icm", {})
        env = icm_wrapper(
            env,
            icm_only=icm_only,
            icm_beta=icm_beta,
            **icm_cfg,
        )
    elif wrapper:
        env = reward_shaping_wrapper(env)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
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

    print(f"\n{algo_name} evaluation results (seed={seed}, {n_episodes} episodes)")
    print(f"  Mean reward : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean length : {stats['mean_length']:.1f} steps")

    return stats


# Main
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
    parser.add_argument(
        "--icm",
        action="store_true",
        help="Apply ICM curiosity wrapper during evaluation.",
    )
    parser.add_argument(
        "--icm_only",
        action="store_true",
        help="Suppress extrinsic reward during evaluation.",
    )
    parser.add_argument(
        "--icm_beta",
        type=float,
        default=1.0,
        help="Weight on intrinsic reward in additive mode (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used during training — selects the matching result directory (default: 42).",
    )
    args = parser.parse_args()
    evaluate(
        args.algo,
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render,
        wrapper=args.wrapper,
        terminal_reward=args.terminal_reward,
        icm=args.icm,
        icm_only=args.icm_only,
        icm_beta=args.icm_beta,
        seed=args.seed,
    )
