import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper that replaces the environment reward with a terminal signal.

    LunarLander observation:
        obs[0] x position  
        obs[1] y position
        obs[2] x velocity
        obs[3] y velocity    
        obs[4] angle          
        obs[5] angular velocity
        obs[6] left leg contact  
        obs[7] right leg contact 

    At episode end:

        +30   proximity to pad   
        +15   left  leg contact
        +15   right leg contact
        -30   tilt penalty         — proportional to |angle|/π
        +20   low final speed      — scales from 0 (speed≥1) to 20 (speed=0)
        +10   soft vertical touch  — scales from 0 (|vy|≥1) to 10 (vy=0)
        -10   angular velocity     — proportional to min(|ω|, 1)
        -20   horizontal drift     — proportional to min(|vx|, 1) at landing
        +100  safe landing bonus   — terminated + both legs on ground
        -100  crash penalty        — terminated + no legs (hit body or side)
        -50   base timeout penalty — truncated without landing
        -x    timeout state penalty— more penalty for far/fast/tilted timeout
        -x    touchdown motion     — penalty if leg contact occurs while still moving fast
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _terminal_reward(self, obs, terminated: bool, truncated: bool) -> float:
        x_pos      = float(obs[0])
        vx         = float(obs[2])
        vy         = float(obs[3])
        angle      = float(obs[4])
        angular_v  = float(obs[5])
        left_leg   = obs[6] > 0.5
        right_leg  = obs[7] > 0.5

        reward = 0.0

        # Proximity to landing pad (x=0 is the centre)
        reward += 30.0 * max(0.0, 1.0 - abs(x_pos))

        # Leg contact
        if left_leg:
            reward += 15.0
        if right_leg:
            reward += 15.0

        # Tilt penalty
        reward -= 30.0 * min(abs(angle), np.pi) / np.pi

        # Low total speed reward
        speed = np.sqrt(vx ** 2 + vy ** 2)
        reward += 20.0 * max(0.0, 1.0 - speed)

        # Soft vertical touchdown
        reward += 30.0 * max(0.0, 1.0 - abs(vy))

        # Angular velocity penalty
        reward -= 10.0 * min(abs(angular_v), 1.0)

        # Horizontal drift penalty at landing
        reward -= 20.0 * min(abs(vx), 1.0)

        # Penalize touching down with too much residual motion.
        # This discourages sliding/spinning.
        touched_ground = left_leg or right_leg
        if touched_ground:
            speed_excess = max(0.0, speed - 0.25)
            ang_vel_excess = max(0.0, abs(angular_v) - 0.20)
            reward -= 40.0 * min(speed_excess, 1.5)
            reward -= 25.0 * min(ang_vel_excess, 1.5)

        if terminated:
            if left_leg and right_leg:
                reward += 100.0   
            else:
                reward -= 100.0   # Crash
        elif truncated:
            # Base timeout cost + state-dependent cost to punish unstable/unfinished states.
            timeout_state_penalty = (
                30.0 * min(abs(x_pos), 1.5)
                + 30.0 * min(speed, 1.5)
                + 20.0 * min(abs(angle), 1.0)
            )
            reward -= 50.0 + timeout_state_penalty

        return reward

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if done:
            reward = self._terminal_reward(obs, terminated, truncated)
        else:
            reward = 0.0

        return obs, reward, terminated, truncated, info


def reward_shaping_wrapper(env: gym.Env) -> gym.Wrapper:
    return RewardShapingWrapper(env)
