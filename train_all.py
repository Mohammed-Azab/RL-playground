"""
train_all.py — Train ALL baseline algorithms on LunarLander sequentially.

Usage:
    python train_all.py
    python train_all.py --algos DQN PPO SAC     # subset
    python train_all.py --timesteps 200000       # quick smoke-test
    python train_all.py --seed 0
    python train_all.py --algos SAC --wrapper --terminal_reward
    python train_all.py --algos SAC --wrapper --shaping_scale 0.05
"""

import argparse
import time

from configs import CONFIGS
from train import train, ALGORITHMS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LunarLander algorithms sequentially (baseline or wrapper variants)."
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=list(CONFIGS),
        choices=list(ALGORITHMS),
        help="Subset of algorithms to train (default: all).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override timesteps for every algorithm.",
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
        help="Apply the custom reward wrapper during training.",
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

    total_start = time.time()
    results: dict[str, str] = {}

    for algo in args.algos:
        start = time.time()
        print(f"\n{'#' * 60}")
        print(f"#  Training {algo}")
        print(f"{'#' * 60}")
        try:
            train(
                algo,
                timesteps=args.timesteps,
                seed=args.seed,
                wrapper=args.wrapper,
                terminal_reward=args.terminal_reward,
                shaping_scale=args.shaping_scale,
            )
            elapsed = time.time() - start
            results[algo] = f"✓ done ({elapsed / 60:.1f} min)"
        except Exception as exc:  # noqa: BLE001
            results[algo] = f"✗ FAILED — {exc}"
            print(f"\n[ERROR] {algo} failed: {exc}\n")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for algo, status in results.items():
        print(f"  {algo:6s}  {status}")
    print(f"\n  Total time: {total_elapsed / 60:.1f} min")
    print(f"{'=' * 60}\n")
    print("Run  python compare.py  to plot the comparison chart.")


if __name__ == "__main__":
    main()
