"""
Shared auxillary functions.
"""

import gym
import torch
import numpy as np
from .multiprocessing_env import SubprocVecEnv
gym.logger.set_level(40)


#########      use GPU      ##########
def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#########      create single gym environment      ##########
def create_env(env_name, max_episode_steps):
    """Create a single gym environment."""
    env = gym.make(env_name)
    env._max_episode_steps = max_episode_steps
    return env


#########      create multiple gym environment      ##########
def make_env(env_name, max_episode_steps=None):
    """Create a gym environment instance when using vectorized environments."""
    def _thunk():
        env = gym.make(env_name)
        if max_episode_steps:
            env._max_episode_steps = max_episode_steps
        return env
    return _thunk

def create_envs(env_name, max_episode_steps, num_envs):
    """Create multiple gym environments."""
    envs = [make_env(env_name, max_episode_steps) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    print(f'Parallel Environments: {envs.num_envs}')
    return envs


#########      process rewards     ##########
def discount(rewards, gamma):
    """Calulate discounted future rewards."""
    discounted_rewards = np.zeros_like(rewards)
    cumulative_rewards = 0.
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

def normalize(rewards):
    """Normalize rewards."""
    mean = np.mean(rewards)
    std = np.std(rewards)
    std = max(1e-8, std) # avoid divide by zero if rewards = 0.
    return (rewards - mean) / (std)


#########      display results      ##########
def moving_average(values, window=100):
    """Calculate moving average over window."""
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')
