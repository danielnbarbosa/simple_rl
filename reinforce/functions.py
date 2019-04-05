"""
Local auxillary functions.
"""

import operator
from functools import reduce
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
    returns, steps = zip(*results)
    # need to gather at least window results before moving average is accurate
    smoothed_returns = moving_average(returns)
    print(f'episode: {len(returns)}',          # specific to this episode
          f'return: {returns[-1]:.2f} |',
          f'avg: {smoothed_returns[-1]:.2f}',  # cummulative
          f'max_avg: {np.max(smoothed_returns):.2f}',
          f'cum_steps: {np.sum(steps)}')


def flatten_rollouts(rollouts, gamma):
    """Return flattened version of rollouts with discounted rewards."""

    def flatten_a(dict_of_arrays):
        """Flatten dict of arrays."""
        return np.concatenate([val for val in dict_of_arrays.values()]).tolist()

    def flatten_t(dict_of_tuples):
        """Flatten dict of tuples."""
        unraveled_dict = [val for val in dict_of_tuples.values()]
        flattened_list = reduce(operator.concat, unraveled_dict)
        return flattened_list

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
