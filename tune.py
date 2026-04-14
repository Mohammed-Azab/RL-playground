"""
tune.py — Optuna hyperparameter search for any algorithm in this project.

Usage:
    python tune.py --algo PPO --n_trials 20 --timesteps 100000
    python tune.py --algo DQN --n_trials 30 --timesteps 150000
    python tune.py --algo PPO_ICM --n_trials 15 --timesteps 100000
    python tune.py --algo SAC --n_trials 20 --timesteps 100000 --visualize
    python tune.py --algo PPO --n_trials 20 --timesteps 100000 --wrapper --terminal_reward

After tuning:
    optuna-dashboard sqlite:///PPO_tuning.db   # live dashboard at localhost:8080
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from configs import CONFIGS
from icm import icm_wrapper
from reward_shaping_wrapper import reward_shaping_wrapper
from train import ALGORITHMS

# ── per-algo search spaces ────────────────────────────────────────────────────

def _sample_ppo(trial: optuna.Trial) -> dict:
    n_steps    = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    if batch_size > n_steps:
        raise optuna.exceptions.TrialPruned()
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps":       n_steps,
        "batch_size":    batch_size,
        "n_epochs":      trial.suggest_categorical("n_epochs", [3, 5, 10, 20]),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 1.0),
        "ent_coef":      trial.suggest_float("ent_coef", 0.0, 0.01),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.4),
        "vf_coef":       trial.suggest_float("vf_coef", 0.25, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
    }


def _sample_a2c(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":        trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps":              trial.suggest_categorical("n_steps", [5, 16, 32, 64, 128]),
        "gamma":                trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "gae_lambda":           trial.suggest_float("gae_lambda", 0.9, 1.0),
        "ent_coef":             trial.suggest_float("ent_coef", 0.0, 0.01),
        "vf_coef":              trial.suggest_float("vf_coef", 0.25, 1.0),
        "max_grad_norm":        trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "normalize_advantage":  trial.suggest_categorical("normalize_advantage", [True, False]),
    }


def _sample_dqn(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":         trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":           trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "batch_size":            trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma":                 trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "tau":                   trial.suggest_float("tau", 0.01, 1.0),
        "train_freq":            trial.suggest_categorical("train_freq", [1, 4, 8]),
        "learning_starts":       trial.suggest_categorical("learning_starts", [0, 1000, 5000]),
        "exploration_fraction":  trial.suggest_float("exploration_fraction", 0.05, 0.3),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
    }


def _sample_sac(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":  trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":    trial.suggest_categorical("buffer_size", [100_000, 500_000, 1_000_000]),
        "batch_size":     trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "tau":            trial.suggest_float("tau", 0.001, 0.05),
        "gamma":          trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "train_freq":     trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "ent_coef":       trial.suggest_categorical("ent_coef", ["auto", 0.1, 0.01, 0.001]),
    }


def _sample_ddpg(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":   trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 100, 256]),
        "tau":           trial.suggest_float("tau", 0.001, 0.05),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "train_freq":    trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [-1, 1, 4]),
    }


def _sample_td3(trial: optuna.Trial) -> dict:
    return {
        **_sample_ddpg(trial),
        "policy_delay": trial.suggest_categorical("policy_delay", [1, 2, 3]),
    }


def _sample_icm(trial: optuna.Trial) -> dict:
    return {
        "feature_dim":     trial.suggest_categorical("icm_feature_dim", [32, 64, 128, 256]),
        "lr":              trial.suggest_float("icm_lr", 1e-4, 1e-2, log=True),
        "eta":             trial.suggest_float("icm_eta", 1e-3, 0.5, log=True),
        "beta":            trial.suggest_float("icm_beta_val", 0.1, 0.9),
        "update_freq":     trial.suggest_categorical("icm_update_freq", [64, 128, 256, 512]),
        "buffer_capacity": trial.suggest_categorical("icm_buffer_capacity", [500, 1000, 5000]),
        "batch_size":      trial.suggest_categorical("icm_batch_size", [32, 64, 128]),
    }


_SAMPLERS = {
    "PPO":     _sample_ppo,
    "PPO_ICM": _sample_ppo,
    "A2C":     _sample_a2c,
    "DQN":     _sample_dqn,
    "SAC":     _sample_sac,
    "DDPG":    _sample_ddpg,
    "TD3":     _sample_td3,
}

# ── environment factory ───────────────────────────────────────────────────────

def _make_env(
    algo_name: str,
    wrapper: bool,
    terminal_reward: bool,
    icm_cfg: dict | None,
) -> Monitor:
    env_id = CONFIGS[algo_name]["env"]
    env = gym.make(env_id)
    if wrapper:
        env = reward_shaping_wrapper(env)
    if icm_cfg:
        env = icm_wrapper(env, **icm_cfg)
    return Monitor(env)


# ── objective ─────────────────────────────────────────────────────────────────

def _objective(
    trial: optuna.Trial,
    algo_name: str,
    n_timesteps: int,
    wrapper: bool,
    terminal_reward: bool,
) -> float:
    sampler_fn = _SAMPLERS[algo_name]
    params = sampler_fn(trial)
    icm_cfg = _sample_icm(trial) if algo_name == "PPO_ICM" else None

    train_env = _make_env(algo_name, wrapper, terminal_reward, icm_cfg)
    eval_env  = _make_env(algo_name, wrapper, terminal_reward, icm_cfg)

    AlgoClass = ALGORITHMS[algo_name]
    model = AlgoClass("MlpPolicy", train_env, verbose=0, **params)

    # Evaluate 5 times throughout training and prune bad trials early
    n_checkpoints = 5
    interval      = n_timesteps // n_checkpoints
    best_mean     = -np.inf

    for checkpoint in range(n_checkpoints):
        model.learn(interval, reset_num_timesteps=(checkpoint == 0))
        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=5, deterministic=True, warn=False
        )
        trial.report(mean_reward, checkpoint)
        best_mean = max(best_mean, mean_reward)
        if trial.should_prune():
            train_env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()

    train_env.close()
    eval_env.close()
    return best_mean


# ── visualisation ─────────────────────────────────────────────────────────────

def _visualize(study: optuna.Study) -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        print("plotly not installed — run: pip install plotly kaleido")
        return

    import optuna.visualization as vis

    os.makedirs("results/tuning_plots", exist_ok=True)

    plots = {
        "optimization_history":  vis.plot_optimization_history(study),
        "param_importances":     vis.plot_param_importances(study),
        "parallel_coordinate":   vis.plot_parallel_coordinate(study),
        "slice":                 vis.plot_slice(study),
    }

    # Only add contour if there are at least 2 params
    if len(study.best_trial.params) >= 2:
        param_names = list(study.best_trial.params.keys())[:2]
        plots["contour"] = vis.plot_contour(study, params=param_names)

    for name, fig in plots.items():
        path = f"results/tuning_plots/{study.study_name}_{name}.html"
        fig.write_html(path)
        print(f"  Saved: {path}")

    print("\nOpen any .html file in your browser to explore interactively.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for any algorithm in this project."
    )
    parser.add_argument(
        "--algo",
        required=True,
        choices=list(ALGORITHMS),
        help="Algorithm to tune.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Training steps per trial (default: 200 000). Use lower values for faster but noisier search.",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name. Defaults to '<ALGO>_tuning'.",
    )
    parser.add_argument(
        "--wrapper",
        action="store_true",
        help="Apply the terminal reward wrapper during tuning.",
    )
    parser.add_argument(
        "--terminal_reward",
        action="store_true",
        help="Use terminal-only reward signal (requires --wrapper).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate HTML visualisation plots after tuning completes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the Optuna TPE sampler (default: 42).",
    )
    args = parser.parse_args()

    if args.terminal_reward and not args.wrapper:
        parser.error("--terminal_reward requires --wrapper.")

    study_name = args.study_name or f"{args.algo}_tuning"
    storage    = f"sqlite:///{study_name}.db"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"\nStudy : {study_name}")
    print(f"DB    : {storage}")
    print(f"Algo  : {args.algo}")
    print(f"Trials: {args.n_trials}  |  Steps/trial: {args.timesteps:,}")
    print(f"\nLive dashboard (run in a separate terminal):")
    print(f"  optuna-dashboard {storage}\n")

    study.optimize(
        lambda trial: _objective(
            trial,
            args.algo,
            args.timesteps,
            args.wrapper,
            args.terminal_reward,
        ),
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    # ── results ───────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Best trial #{study.best_trial.number}")
    print(f"  Mean reward : {study.best_trial.value:.2f}")
    print(f"{'=' * 55}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"\nTo paste into configs.py → '{args.algo}' → 'hyperparams'")
    print(f"To explore: optuna-dashboard {storage}")

    if args.visualize:
        print("\nGenerating visualisation plots...")
        _visualize(study)


if __name__ == "__main__":
    main()
