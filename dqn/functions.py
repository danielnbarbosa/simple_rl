"""
Auxillary functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def moving_average(values, window=20):
    """Calculate moving average over window."""
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')


def plot_results(results):
    """Plot training results."""
    rewards, epsilons, buffer_lens = zip(*results)
    plt.figure(figsize=(20, 5))

    plt.subplot(131)
    plt.title('return')
    plt.plot(moving_average(rewards))

    plt.subplot(132)
    plt.title('epsilon')
    plt.plot(moving_average(epsilons))

    plt.subplot(133)
    plt.title('buffer_len')
    plt.plot(moving_average(buffer_lens))

    plt.show()
    print(f'Episode: {len(rewards)}  |  Return: {rewards[-1]}  Epsilon: {epsilons[-1]:.2f}  Buffer Length: {buffer_lens[-1]}')


def print_results(results):
    """Plot training results."""
    rewards, epsilons, buffer_lens = zip(*results)
    print(f'Episode: {len(rewards)}  |  Return: {rewards[-1]}  Epsilon: {epsilons[-1]:.2f}  Buffer Length: {buffer_lens[-1]}')
