"""
icm.py — Intrinsic Curiosity Module (Pathak et al., 2017) as a gym.Wrapper.

The wrapper augments the reward signal with an intrinsic curiosity bonus
derived from forward model prediction error in learned feature space.

Three networks are trained jointly:
  - Encoder       φ: obs → feature vector
  - InverseModel  g: (z_t, z_{t+1}) → predicted action (trains φ to be action-predictive)
  - ForwardModel  f: (z_t, a_vec) → predicted z_{t+1}

For discrete action spaces, a_vec is a one-hot vector.
For continuous action spaces, a_vec is the raw action vector.

Inverse model loss:
  - Discrete:   CrossEntropy(predicted_logits, action_index)
  - Continuous: MSE(predicted_action, actual_action)

Intrinsic reward: r_i = eta/2 * ||f(z_t, a_vec) - z_{t+1}||^2
ICM loss:         L = (1 - beta) * L_inv + beta * L_fwd
Total reward:     r = r_extrinsic + icm_beta * r_i  (or r_i only if icm_only=True)
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class _Encoder(nn.Module):
    def __init__(self, obs_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _InverseModel(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))


class _ForwardModel(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

    def forward(self, z_t: torch.Tensor, a_vec: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, a_vec], dim=-1))


class ICMWrapper(gym.Wrapper):
    """Gym wrapper that adds ICM-based intrinsic curiosity reward.

    Supports both Discrete and Box (continuous) action spaces.

    Args:
        env:              Wrapped environment.
        feature_dim:      Encoder output dimension.
        lr:               Adam learning rate for ICM networks.
        eta:              Intrinsic reward scaling factor.
        beta:             Forward/inverse loss balance (λ in paper, 0=all inverse, 1=all forward).
        icm_beta:         Weight on intrinsic reward when adding to extrinsic (icm_only=False).
        icm_only:         If True, suppress extrinsic reward entirely.
        update_freq:      Number of env steps between ICM gradient updates.
        buffer_capacity:  Max transitions stored in the replay buffer.
        batch_size:       Minibatch size for each ICM update.
    """

    def __init__(
        self,
        env: gym.Env,
        feature_dim: int = 64,
        lr: float = 1e-3,
        eta: float = 0.01,
        beta: float = 0.2,
        icm_beta: float = 1.0,
        icm_only: bool = False,
        update_freq: int = 128,
        buffer_capacity: int = 1000,
        batch_size: int = 64,
    ):
        super().__init__(env)

        obs_dim = int(np.prod(env.observation_space.shape))

        self._discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self._discrete:
            action_dim = int(env.action_space.n)
        else:
            assert isinstance(env.action_space, gym.spaces.Box), (
                "ICMWrapper supports Discrete and Box action spaces only."
            )
            action_dim = int(np.prod(env.action_space.shape))

        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.eta = eta
        self.beta = beta
        self.icm_beta = icm_beta
        self.icm_only = icm_only
        self.update_freq = update_freq
        self.batch_size = batch_size

        # Networks
        self.encoder = _Encoder(obs_dim, feature_dim)
        self.inverse_model = _InverseModel(feature_dim, action_dim)
        self.forward_model = _ForwardModel(feature_dim, action_dim)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.inverse_model.parameters())
            + list(self.forward_model.parameters()),
            lr=lr,
        )

        # Replay buffer: stores (obs, action, obs_next) tuples
        self._buffer: deque = deque(maxlen=buffer_capacity)
        self._obs_prev: np.ndarray | None = None
        self._step_count: int = 0

        # Stats exposed to TensorBoard callback
        self.last_forward_loss: float = 0.0
        self.last_inverse_loss: float = 0.0
        self.last_icm_loss: float = 0.0
        self.last_intrinsic_reward: float = 0.0
        self._ep_intrinsic_reward: float = 0.0
        self.last_ep_intrinsic_reward: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_prev = obs.copy()
        self._step_count = 0
        self.last_ep_intrinsic_reward = self._ep_intrinsic_reward
        self._ep_intrinsic_reward = 0.0
        return obs, info

    def step(self, action):
        obs_next, r_e, terminated, truncated, info = self.env.step(action)

        r_i = self._intrinsic_reward(self._obs_prev, action, obs_next)
        self.last_intrinsic_reward = r_i
        self._ep_intrinsic_reward += r_i
        self._buffer.append((self._obs_prev.copy(), np.array(action, dtype=np.float32), obs_next.copy()))
        self._step_count += 1

        if self._step_count % self.update_freq == 0 and len(self._buffer) >= self.batch_size:
            self._update_icm()

        if self.icm_only:
            reward = float(r_i)
        else:
            reward = float(r_e) + self.icm_beta * float(r_i)

        self._obs_prev = obs_next.copy()
        return obs_next, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _action_vec(self, action) -> torch.Tensor:
        """Return a (1, action_dim) float tensor for a single action."""
        if self._discrete:
            a = torch.zeros(1, self.action_dim)
            a[0, int(action)] = 1.0
        else:
            a = torch.FloatTensor(np.array(action, dtype=np.float32).flatten()).unsqueeze(0)
        return a

    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        return self.encoder(torch.FloatTensor(obs).unsqueeze(0))

    def _intrinsic_reward(self, obs: np.ndarray, action, obs_next: np.ndarray) -> float:
        with torch.no_grad():
            z_t = self._encode(obs)
            z_next = self._encode(obs_next)
            a_vec = self._action_vec(action)
            z_next_pred = self.forward_model(z_t, a_vec)
            r_i = (self.eta / 2.0) * F.mse_loss(z_next_pred, z_next, reduction="sum").item()
        return r_i

    def _update_icm(self) -> None:
        batch = random.sample(self._buffer, self.batch_size)
        obs_b, actions_b, obs_next_b = zip(*batch)

        obs_t = torch.FloatTensor(np.array(obs_b))
        obs_next_t = torch.FloatTensor(np.array(obs_next_b))
        actions_arr = np.array(actions_b, dtype=np.float32)

        if self._discrete:
            # actions_arr shape: (batch,)  — scalar indices stored as float
            actions_idx = torch.LongTensor(actions_arr.astype(np.int64))
            a_vec = torch.zeros(self.batch_size, self.action_dim)
            a_vec.scatter_(1, actions_idx.unsqueeze(1), 1.0)
        else:
            # actions_arr shape: (batch, action_dim)
            a_vec = torch.FloatTensor(actions_arr.reshape(self.batch_size, self.action_dim))

        z_t = self.encoder(obs_t)
        z_next = self.encoder(obs_next_t)

        # Forward loss
        z_next_pred = self.forward_model(z_t.detach(), a_vec)
        l_fwd = 0.5 * F.mse_loss(z_next_pred, z_next.detach())

        # Inverse loss
        a_pred = self.inverse_model(z_t, z_next)
        if self._discrete:
            l_inv = F.cross_entropy(a_pred, actions_idx)
        else:
            l_inv = F.mse_loss(a_pred, a_vec)

        loss = (1.0 - self.beta) * l_inv + self.beta * l_fwd

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_forward_loss = l_fwd.item()
        self.last_inverse_loss = l_inv.item()
        self.last_icm_loss = loss.item()


def icm_wrapper(
    env: gym.Env,
    feature_dim: int = 64,
    lr: float = 1e-3,
    eta: float = 0.01,
    beta: float = 0.2,
    icm_beta: float = 1.0,
    icm_only: bool = False,
    update_freq: int = 128,
    buffer_capacity: int = 1000,
    batch_size: int = 64,
) -> ICMWrapper:
    """Factory function matching the style of reward_shaping_wrapper.py."""
    return ICMWrapper(
        env,
        feature_dim=feature_dim,
        lr=lr,
        eta=eta,
        beta=beta,
        icm_beta=icm_beta,
        icm_only=icm_only,
        update_freq=update_freq,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
    )
