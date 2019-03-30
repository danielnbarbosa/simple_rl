"""
Auxillary functions.
"""

import torch
import numpy as np


def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def moving_average(values, window=20):
    """Calculate moving average over window."""
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')


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

    print(f'episode: {i_episode} return: {i_ret:.2f} eps: {i_eps:.2f} buff: {i_buffer_len} avg: {i_avg_ret:.2f} | max_avg: {max_avg_ret:.2f}')
