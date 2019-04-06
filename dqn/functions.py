"""
Functions specific to DQN.
"""

import numpy as np
from common.misc import moving_average
from .models import MLP, CNN


def create_mlp(device, env):
    """Create MLP models based on environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    q_net = MLP(state_size, action_size).to(device)
    target_net = MLP(state_size, action_size).to(device)
    return (q_net, target_net)


def create_cnn(device, action_size, frames=4):
    """Create CNN models."""
    q_net = CNN(frames, action_size).to(device)
    target_net = CNN(frames, action_size).to(device)
    return (q_net, target_net)


def print_results(results):
    """Print results."""
    returns, epsilons, buffer_lens, steps = zip(*results)
    # need to gather at least window results before moving average is accurate
    smoothed_returns = moving_average(returns)
    print(f'episode: {len(returns)}',          # specific to this episode
          f'return: {returns[-1]:.2f}',
          f'eps: {epsilons[-1]:.2f}',
          f'buff: {buffer_lens[-1]} |',
          f'avg: {smoothed_returns[-1]:.2f}',  # cummulative
          f'max_avg: {np.max(smoothed_returns):.2f}',
          f'cum_steps: {np.sum(steps)}')
