"""
train_all.py — Train ALL baseline algorithms on LunarLander sequentially.

Usage:
    python train_all.py
    python train_all.py --algos DQN PPO SAC     # subset
    python train_all.py --timesteps 200000       # quick smoke-test
    python train_all.py --seed 0
"""

import argparse
import time

from configs import CONFIGS
from train import train, ALGORITHMS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all LunarLander baselines sequentially."
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
    args = parser.parse_args()

    total_start = time.time()
    results: dict[str, str] = {}

    for algo in args.algos:
        start = time.time()
        print(f"\n{'#' * 60}")
        print(f"#  Training {algo}")
        print(f"{'#' * 60}")
        try:
            train(algo, timesteps=args.timesteps, seed=args.seed)
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
