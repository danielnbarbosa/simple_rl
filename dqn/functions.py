"""
Auxillary functions.
"""

import gym
import torch
import numpy as np
from models import TwoLayerMLP


def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(env_name, max_episode_steps):
    env = gym.make(env_name)
    env._max_episode_steps = 1000
    return env


def create_models(env, hidden_size):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = get_device()
    q_net = TwoLayerMLP((state_size, *hidden_size, action_size)).to(device)
    target_net = TwoLayerMLP((state_size, *hidden_size, action_size)).to(device)
    return (q_net, target_net)


def moving_average(values, window=100):
    """Calculate moving average over window."""
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')


def print_results(results):
    """Print results."""
    returns, epsilons, buffer_lens = zip(*results)
    smoothed_returns = moving_average(returns)  # need to gather at least window results before this is accurate
    i_episode = len(returns)
    i_ret = returns[-1]
    i_eps = epsilons[-1]
    i_buffer_len = buffer_lens[-1]
    i_avg_ret = smoothed_returns[-1]
    max_avg_ret = np.max(smoothed_returns)

    print(f'episode: {i_episode} return: {i_ret:.2f} eps: {i_eps:.2f} buff: {i_buffer_len} avg: {i_avg_ret:.2f} | max_avg: {max_avg_ret:.2f}')
