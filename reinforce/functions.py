"""
Functions specific to REINFORCE.
"""

import numpy as np
from common.misc import moving_average
from .models import MLP, CNN


def create_mlp(device, env):
    """Create MLP model based on an environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    return MLP(state_size, action_size).to(device)


def create_cnn(device, action_size, frames=4):
    """Create CNN models."""
    return CNN(frames, action_size).to(device)


def print_results(results):
    """Print results.  Pipe separates episode specific from cummulative results."""
    returns, steps = zip(*results)
    smoothed_returns = moving_average(returns)
    print(f'episode: {len(returns)}',
          f'return: {returns[-1]:.2f} |',
          f'avg: {smoothed_returns[-1]:.2f}',
          f'max_avg: {np.max(smoothed_returns):.2f}',
          f'cum_steps: {np.sum(steps)}')


def unzip_rollouts(rollouts):
    """Unzip a dictionary of rollouts into dictionaries of their underlying values."""
    num_envs = len(rollouts)
    # create dictionaries indexed by environment id
    rewards = {n: None for n in range(num_envs)}
    log_probs = {n: None for n in range(num_envs)}
    # populate dictonaries with unzipped tuples in rollouts
    for n in range(num_envs):
        rewards[n], log_probs[n] = zip(*rollouts[n])
    return rewards, log_probs
