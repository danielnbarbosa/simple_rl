"""
Local auxillary functions.
"""

import numpy as np
from .models import TwoLayerMLP
from common.functions import get_device, moving_average


def create_models(env):
    """Create models based on an environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = get_device()
    q_net = TwoLayerMLP(state_size, action_size).to(device)
    target_net = TwoLayerMLP(state_size, action_size).to(device)
    return (q_net, target_net)


def print_results(results):
    """Print results."""
    returns, epsilons, buffer_lens = zip(*results)
    smoothed_returns = moving_average(returns)
    i_episode = len(returns)
    i_ret = returns[-1]
    i_eps = epsilons[-1]
    i_buffer_len = buffer_lens[-1]
    i_avg_ret = smoothed_returns[-1]
    max_avg_ret = np.max(smoothed_returns)
    # need to gather at least window results before moving average is accurate
    print(f'episode: {i_episode} return: {i_ret:.2f} eps: {i_eps:.2f} buff: {i_buffer_len} avg: {i_avg_ret:.2f} | max_avg: {max_avg_ret:.2f}')
