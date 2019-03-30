import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
gym.logger.set_level(40)


def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_env(env_name, max_episode_steps=None):
    """Create a gym environment when using vectorized environments."""
    def _thunk():
        env = gym.make(env_name)
        if max_episode_steps:
            env._max_episode_steps = max_episode_steps
        return env
    return _thunk


def moving_average(values, window=20):
    """Calculate moving average over window."""
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')


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


def plot_results(all_rewards):
    """Plot results."""
    clear_output(True)
    plt.figure(figsize=(6, 5))
    plt.subplot(111)
    plt.title('return')
    plt.plot(moving_average(all_rewards))
    plt.show()
    print(f'Episode: {len(all_rewards)}  |  Return: {all_rewards[-1]}')


def print_results(all_rewards):
    """Print results."""
    print(f'Episode: {len(all_rewards)}  |  Return: {all_rewards[-1]}  Avg: {moving_average(all_rewards)[-1]:.2f}')


def flatten_a(values):
    """Flatten a dict of arrays."""
    return np.concatenate([val for val in values.values()])


def flatten_t(values):
    """Flatten a dict of tensors."""
    for n in range(len(values)):
        values[n] = torch.stack(values[n], dim=0)
    return torch.cat([val for val in values.values()])


def flatten_rollouts(rollouts, gamma):
    """Return flattened version of rollouts with discounted rewards."""
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
