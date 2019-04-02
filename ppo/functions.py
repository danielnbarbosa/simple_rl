"""
Local auxillary functions.
"""

import torch
import numpy as np
from .models import TwoLayerMLP
from common.functions import get_device, moving_average, discount


def create_model(env):
    """Create a model based on an environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = get_device()
    model = TwoLayerMLP(state_size, action_size).to(device)
    return model


def print_results(returns, epsilons):
    """Print results."""
    smoothed_returns = moving_average(returns)
    i_episode = len(returns)
    i_ret = returns[-1]
    i_eps = epsilons[-1]
    i_avg_ret = smoothed_returns[-1]
    max_avg_ret = np.max(smoothed_returns)
    # need to gather at least window results before moving average is accurate
    print(f'episode: {i_episode} return: {i_ret:.2f} eps: {i_eps:.2f} avg: {i_avg_ret:.2f} | max_avg: {max_avg_ret:.2f}')


def flatten_rollouts(rollouts, gamma):
    """Return flattened version of rollouts with discounted rewards."""

    def flatten_a(values):
        """Flatten a dict of arrays."""
        return np.concatenate([val for val in values.values()])

    num_envs = len(rollouts)
    # create dictionaries indexed by agent id
    rewards = {n: None for n in range(num_envs)}
    discounted_rewards = {n: None for n in range(num_envs)}
    probs = {n: None for n in range(num_envs)}
    states = {n: None for n in range(num_envs)}
    actions = {n: None for n in range(num_envs)}

    for n in range(num_envs):
        rewards[n], probs[n], states[n], actions[n] = zip(*rollouts[n])
        # discount rewards across each rollout
        discounted_rewards[n] = discount(rewards[n], gamma)

    return flatten_a(rewards), flatten_a(discounted_rewards), flatten_a(probs), flatten_a(states), flatten_a(actions)
