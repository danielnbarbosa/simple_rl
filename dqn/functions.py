"""
Local auxillary functions.
"""

import numpy as np
from .models import TwoLayerMLP, ConvNet
from common.functions import get_device, moving_average


def create_mlp_models(env):
    """Create MLP models based on environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = get_device()
    q_net = TwoLayerMLP(state_size, action_size).to(device)
    target_net = TwoLayerMLP(state_size, action_size).to(device)
    return (q_net, target_net)


def create_cnn_models(frames, action_size):
    """Create CNN models."""
    device = get_device()
    q_net = ConvNet(frames, action_size).to(device)
    target_net = ConvNet(frames, action_size).to(device)
    return (q_net, target_net)


def print_results(results):
    """Print results."""
    returns, epsilons, buffer_lens, steps = zip(*results)
    smoothed_returns = moving_average(returns)
    # need to gather at least window results before moving average is accurate
    print(f'episode: {len(returns)}',             # specific to this episode
          f'return: {returns[-1]:.2f}',
          f'eps: {epsilons[-1]:.2f}',
          f'buff: {buffer_lens[-1]} |',
          f'avg: {smoothed_returns[-1]:.2f}',     # cummulative
          f'max_avg: {np.max(smoothed_returns):.2f}',
          f'cum_steps: {np.sum(steps)}')
