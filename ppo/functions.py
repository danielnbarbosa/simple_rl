"""
Local auxillary functions.
"""

import numpy as np
from common.functions import moving_average, discount
from .models import MLP


def create_mlp(device, env):
    """Create MLP model based on an environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    return MLP(state_size, action_size).to(device)


def print_results(results):
    """Print results."""
    returns, epsilons, steps = zip(*results)
    # need to gather at least window results before moving average is accurate
    smoothed_returns = moving_average(returns)
    print(f'episode: {len(returns)}',          # specific to this episode
          f'return: {returns[-1]:.2f}',
          f'eps: {epsilons[-1]:.2f} |',
          f'avg: {smoothed_returns[-1]:.2f}',  # cummulative
          f'max_avg: {np.max(smoothed_returns):.2f}',
          f'cum_steps: {np.sum(steps)}')


def unzip_rollouts(rollouts):
    """Unzip a dictionary of rollouts into dictionaries of their underlying values."""
    num_envs = len(rollouts)
    # create dictionaries indexed by environment id
    rewards = {n: None for n in range(num_envs)}
    probs = {n: None for n in range(num_envs)}
    states = {n: None for n in range(num_envs)}
    actions = {n: None for n in range(num_envs)}
    # populate dictonaries with unzipped tuples in rollouts
    for n in range(num_envs):
        rewards[n], probs[n], states[n], actions[n] = zip(*rollouts[n])
    return rewards, probs, states, actions
