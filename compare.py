"""
compare.py — Load training logs for every trained algorithm and plot a
             side-by-side reward comparison.

The script reads the ``monitor.csv`` files written by stable-baselines3's
Monitor wrapper (one per algorithm) from ``results/<ALGO>/logs/``.

Usage:
    python compare.py                         # compare all trained algos
    python compare.py --algos DQN PPO A2C     # subset
    python compare.py --window 20             # smoothing window (episodes)
    python compare.py --output my_plot.png    # custom output filename
    python compare.py --no_save               # only display, don't save
"""

import os
import argparse
import glob as _glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from configs import CONFIGS


# Color palette — one color per algorithm (up to 6)
_PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#6A4C93"]


def _load_monitor_csv(log_dir: str) -> pd.DataFrame | None:
    """Return a DataFrame with cumulative *steps* and *episode reward* columns,
    or ``None`` if no monitor file is found in *log_dir*."""
    # stable-baselines3 Monitor creates a file named ``<prefix>.monitor.csv``
    # or just ``monitor.csv`` depending on the filename argument.
    patterns = [
        os.path.join(log_dir, "*.monitor.csv"),
        os.path.join(log_dir, "monitor.csv"),
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(_glob.glob(pat))

    if not files:
        return None

    # If multiple shards exist (parallel envs), concatenate them.
    frames: list[pd.DataFrame] = []
    for fp in sorted(files):
        try:
            df = pd.read_csv(fp, comment="#")
            frames.append(df)
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    # Monitor CSV columns: r (reward), l (length), t (wall time seconds)
    df = df.rename(columns={"r": "reward", "l": "length", "t": "time"})
    df["cumsteps"] = df["length"].cumsum()
    return df[["cumsteps", "reward"]].copy()


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Return a rolling-mean smoothed version of *values*."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    pad = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def compare(
    algos: list[str] | None = None,
    window: int = 20,
    output: str = "results/comparison.png",
    no_save: bool = False,
) -> None:
    """Plot a reward-vs-timesteps comparison for all (or the given) algorithms.

    Args:
        algos:   List of algorithm names.  ``None`` → use all entries in
                 CONFIGS for which a monitor log exists.
        window:  Episode-level rolling-average window for smoothing.
        output:  File path to save the resulting figure.
        no_save: If ``True``, display the figure interactively instead of
                 saving it.
    """
    if algos is None:
        algos = list(CONFIGS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    ax_disc = axes[0]  # discrete-action algorithms
    ax_cont = axes[1]  # continuous-action algorithms

    ax_disc.set_title("Discrete-Action Algorithms\n(LunarLander-v3)", fontsize=13)
    ax_cont.set_title(
        "Continuous-Action Algorithms\n(LunarLanderContinuous-v3)", fontsize=13
    )

    discrete_algos = [a for a in algos if CONFIGS[a]["env"] == "LunarLander-v3"]
    continuous_algos = [
        a for a in algos if CONFIGS[a]["env"] == "LunarLanderContinuous-v3"
    ]

    found_any = False
    for group, ax, color_offset in [
        (discrete_algos, ax_disc, 0),
        (continuous_algos, ax_cont, 3),
    ]:
        for i, algo in enumerate(group):
            log_dir = os.path.join("results", algo, "logs")
            df = _load_monitor_csv(log_dir)
            if df is None or df.empty:
                print(
                    f"  [skip] No monitor log found for {algo} in '{log_dir}'. "
                    f"Run: python train.py --algo {algo}"
                )
                continue

            found_any = True
            x = df["cumsteps"].to_numpy()
            y = df["reward"].to_numpy()
            y_smooth = _smooth(y, window)
            color = _PALETTE[(color_offset + i) % len(_PALETTE)]

            ax.plot(x, y, alpha=0.15, color=color, linewidth=0.8)
            ax.plot(x, y_smooth, label=algo, color=color, linewidth=2.0)

    if not found_any:
        print(
            "\nNo training logs found. "
            "Run 'python train_all.py' first, then re-run compare.py.\n"
        )
        plt.close(fig)
        return

    for ax in axes:
        ax.axhline(200, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="Solved (≥200)")
        ax.set_xlabel("Training Timesteps", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6
                                  else f"{v / 1e3:.0f}K")
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("LunarLander — RL Baseline Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if no_save:
        plt.show()
    else:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"\n✓ Comparison plot saved to: {output}\n")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare trained LunarLander agents."
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=None,
        choices=list(CONFIGS),
        help="Subset of algorithms to include (default: all trained).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling-average smoothing window in episodes (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.png",
        help="Output file path for the plot (default: results/comparison.png).",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Display the plot interactively instead of saving it.",
    )
    args = parser.parse_args()
    compare(
        algos=args.algos,
        window=args.window,
        output=args.output,
        no_save=args.no_save,
    )
