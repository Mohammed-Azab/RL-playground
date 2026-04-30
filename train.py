import os
import argparse
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO, A2C, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from configs import CONFIGS
from reward_shaping_wrapper import reward_shaping_wrapper
from icm import icm_wrapper
from callbacks import ICMLoggingCallback

ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3,
    "PPO_ICM": PPO,
}

def train(
    algo_name: str,
    timesteps: int | None = None,
    seed: int = 42,
    wrapper: bool = False,
    terminal_reward: bool = False,
    device: str = "auto",
    tensorboard_log: str = "tensorboard_logs",
) -> None:
    """Train *algo_name* on the environment specified in CONFIGS and save the
    trained model plus episode logs to ``results/<algo_name>/``.

    Args:
        algo_name: One of DQN | PPO | A2C | SAC | DDPG | TD3 | PPO_ICM.
        timesteps:  Override the default number of training timesteps.
        seed:       Random seed for reproducibility.
        wrapper:    Whether to apply the custom reward wrapper.
        terminal_reward: Use terminal reward only instead of dense shaping.
        device:     Device selection: auto | cpu | cuda.
    """
    if algo_name not in CONFIGS:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Choose from: {', '.join(CONFIGS)}"
        )
    if terminal_reward and not wrapper:
        raise ValueError("--terminal_reward requires --wrapper.")

    icm = algo_name == "PPO_ICM"

    # Seed RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # reducing nondeterminism because the training gets slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = CONFIGS[algo_name]
    env_id = config["env"]
    hyperparams = config["hyperparams"].copy()
    policy = hyperparams.pop("policy")
    icm_cfg = CONFIGS[algo_name].get("icm", {}) if icm else {}
    n_timesteps = timesteps if timesteps is not None else config["timesteps"]
    run_variant = _run_variant_name(wrapper, terminal_reward, icm, False)
    resolved_device = _resolve_device(device)

    # Directory setup
    log_dir, model_dir = _result_dirs(algo_name, run_variant, seed)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Printing Training Info
    print(f"\n{'=' * 60}")
    print(f"  Algorithm : {algo_name}")
    print(f"  Env       : {env_id}")
    print(f"  Timesteps : {n_timesteps:,}")
    print(f"  Seed      : {seed}")
    print(f"  Variant   : {run_variant}")
    print(f"  Device    : requested={device} resolved={resolved_device}")
    print(f"  Torch     : {torch.__version__} (CUDA build: {torch.version.cuda})")
    print(f"  CUDA      : available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    if resolved_device == "cuda" and torch.cuda.is_available():
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    if wrapper:
        print(f"  Wrapper   : enabled")
        print(f"  TermOnly  : {terminal_reward}")
    if icm:
        print(f"  ICM       : enabled")
    print(f"  Policy    : {policy}")
    _print_param_block("Hyperparams", hyperparams)
    if icm:
        _print_param_block("ICM params", icm_cfg)
    print(f"{'=' * 60}\n")


    # Environments
    train_base_env = gym.make(env_id)
    eval_base_env = gym.make(env_id)

    train_base_env.reset(seed=seed)
    eval_base_env.reset(seed=seed + 123456)

    # Seed the action and observation spaces
    train_base_env.action_space.seed(seed)
    train_base_env.observation_space.seed(seed)
    eval_base_env.action_space.seed(seed + 123456)
    eval_base_env.observation_space.seed(seed + 123456)

    if icm:
        if wrapper and terminal_reward:
            train_base_env = reward_shaping_wrapper(train_base_env)
            eval_base_env = reward_shaping_wrapper(eval_base_env)
        train_base_env = icm_wrapper(
            train_base_env,
            **icm_cfg,
        )
        eval_base_env = icm_wrapper(
            eval_base_env,
            **icm_cfg,
        )
    elif wrapper:
        train_base_env = reward_shaping_wrapper(train_base_env)
        eval_base_env = reward_shaping_wrapper(eval_base_env)

    train_env = Monitor(train_base_env, log_dir)
    eval_env = Monitor(eval_base_env)

    # Model
    AlgoClass = ALGORITHMS[algo_name]
    model = AlgoClass(
        policy,
        train_env,
        seed=seed,
        verbose=1,
        device=resolved_device,
        tensorboard_log=tensorboard_log,
        **hyperparams,
    )

    # Callbacks
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
    callbacks = [eval_callback, checkpoint_callback]
    if icm:
        callbacks.append(ICMLoggingCallback())

    # Training
    tb_log_name = f"{algo_name}_{run_variant}_s{seed}"
    model.learn(
        total_timesteps=n_timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name,
    )

    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\n✓ {algo_name} training complete.")
    print(f"  Model  → {final_path}.zip")
    print(f"  Logs   → {log_dir}/\n")

    train_env.close()
    eval_env.close()

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


def _result_dirs(algo_name: str, run_variant: str, seed: int) -> tuple[str, str]:
    folder = f"{run_variant}_s{seed}"
    base_dir = os.path.join("results", algo_name, folder)
    return os.path.join(base_dir, "logs"), os.path.join(base_dir, "models")


def _resolve_device(requested_device: str) -> str:
    requested = requested_device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but CUDA is not available. "
                "Use --device cpu or --device auto."
            )
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_param_block(title: str, params: dict) -> None:
    print(f"  {title}:")
    for key in sorted(params):
        print(f"    - {key}: {params[key]}")


# Main
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device: auto (default), cpu, or cuda.",
    )
    parser.add_argument(
        "--tensorboard_log",
        type=str,
        default="tensorboard_logs",
        help="Root directory for TensorBoard logs (default: tensorboard_logs).",
    )
    args = parser.parse_args()
    train(
        args.algo,
        timesteps=args.timesteps,
        seed=args.seed,
        wrapper=args.wrapper,
        terminal_reward=args.terminal_reward,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
    )
