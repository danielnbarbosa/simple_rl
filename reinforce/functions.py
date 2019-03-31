"""
Auxillary functions.
"""

import torch
import numpy as np
import gym
from models import TwoLayerMLP
from multiprocessing_env import SubprocVecEnv


def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(env_name, max_episode_steps):
    """Create a single gym environment."""
    env = gym.make(env_name)
    env._max_episode_steps = 1000
    return env


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


def create_model(env, hidden_size):
    """Create a model based on an environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = get_device()
    model = TwoLayerMLP((state_size, *hidden_size, action_size)).to(device)
    return model


def moving_average(values, window=100):
    """Calculate moving average over window."""
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')


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


def print_results(returns):
    """Print results."""
    smoothed_returns = moving_average(returns)
    i_episode = len(returns)
    i_ret = returns[-1]
    i_avg_ret = smoothed_returns[-1]
    max_avg_ret = np.max(smoothed_returns)

    print(f'episode: {i_episode} return: {i_ret:.2f} avg: {i_avg_ret:.2f} | max_avg: {max_avg_ret:.2f}')


def flatten_a(values):
    """Flatten a dict of arrays."""
    return np.concatenate([val for val in values.values()])


def flatten_t(values):
    """Flatten a dict of tensors."""
    for n in range(len(values)):
        values[n] = torch.stack(values[n], dim=0)
    return torch.cat([val for val in values.values()])


def flatten_rollouts(rollouts, gamma):
    """Return flattened version of rollouts with discounted rewards."""
    num_envs = len(rollouts)
    # create dictionaries indexed by agent id
    rewards = {n: None for n in range(num_envs)}
    discounted_rewards = {n: None for n in range(num_envs)}
    log_probs = {n: None for n in range(num_envs)}

    for n in range(num_envs):
        rewards[n], log_probs[n] = zip(*rollouts[n])
        # discount rewards across each rollout
        discounted_rewards[n] = discount(rewards[n], gamma)

    return flatten_a(rewards), flatten_a(discounted_rewards), flatten_t(log_probs)
