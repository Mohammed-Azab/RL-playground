import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    """Gym wrapper that applies terminal reward with or without dense shaping."""

    def __init__(self, env: gym.Env, shaping_scale: float = 0.05):
        super().__init__(env)
        self.shaping_scale = shaping_scale
        self._episode_reward = 0.0

    def reset(self, **kwargs):
        self._episode_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += float(original_reward)

        done = terminated or truncated
        reward = 0.0

        if done:
            if self._episode_reward >= 200.0:
                reward = 100.0
            else:
                reward = -100.0
            self._episode_reward = 0.0

        x, y, vx, vy, angle, _angular_vel, left_leg, right_leg = obs

        distance_weight = 0.5
        vel_weight = 0.2
        angle_weight = 0.2
        leg_weight = 10

        d = distance_weight * np.sqrt(x**2 + y**2)
        v_pen = vel_weight * np.sqrt(vx**2 + vy**2)
        theta = angle_weight * abs(angle)

        shaping = -(d + v_pen + theta)

        # Legs are worth more since they are crucial for landing
        shaping += leg_weight * (left_leg + right_leg)

        reward += self.shaping_scale * shaping

        return obs, reward, terminated, truncated, info


def reward_shaping_wrapper(env: gym.Env, shaping_scale: float = 0.05) -> gym.Wrapper:
    return RewardShapingWrapper(env, shaping_scale=shaping_scale)
