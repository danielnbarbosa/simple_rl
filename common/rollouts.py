"""
Functions for processing rollouts in policy gradient methods.
"""

import operator
from functools import reduce
import numpy as np


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


def discount_and_flatten_rewards(rewards, gamma):
    """Discount rewards across each rollout then flatten all rollouts."""
    num_envs = len(rewards)
    # create dictionary indexed by environment id
    discounted_rewards = {n: None for n in range(num_envs)}
    for n in range(num_envs):
        discounted_rewards[n] = discount(rewards[n], gamma)
    discounted_rewards = list(discounted_rewards.values())
    return np.concatenate(discounted_rewards)


def flatten(values):
    """Flatten a dictionary of values."""
    unraveled = list(values.values())
    flattened = reduce(operator.concat, unraveled)
    return flattened
