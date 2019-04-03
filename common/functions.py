"""
Shared auxillary functions.
"""

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .multiprocessing_env import SubprocVecEnv
gym.logger.set_level(40)


#########      GPU      ##########
def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#########      environments      ##########
def create_env(env_name, max_episode_steps):
    """Create a single gym environment."""
    env = gym.make(env_name)
    env._max_episode_steps = max_episode_steps
    return env

def make_env(env_name, max_episode_steps=None):
    """Create a gym environment instance when using vectorized environments."""
    def _thunk():
        env = gym.make(env_name)
        if max_episode_steps:
            env._max_episode_steps = max_episode_steps
        return env
    return _thunk

def create_envs(env_name, max_episode_steps, num_envs):
    """Create multiple gym environments."""
    envs = [make_env(env_name, max_episode_steps) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    print(f'Parallel Environments: {envs.num_envs}')
    return envs


#########      rewards     ##########
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


#########      results      ##########
def moving_average(values, window=100):
    """Calculate moving average over window."""
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')


#########      atari       ##########
def preprocess_frame(frame):
    """
    Pre-process a single Atari game frame.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert to gray scale
    frame = cv2.resize(frame, (84, 84))             # squish
    frame = frame / 255                             # normalize
    return frame

def env_reset_with_frames(env, n_frames):
    """Reset environment and generate multiple noop frames."""
    frames = []
    frame = env.reset()
    frame = preprocess_frame(frame)
    frames.append(frame)
    # 0 is NOOP so the agent doesn't get to move during these frames
    for i in range(n_frames-1):
        frame, _, _, _ = env.step(0)
        frame = preprocess_frame(frame)
        frames.append(frame)
    state = np.expand_dims(np.asarray(frames), 0)
    return state

def env_step_with_frames(env, action, n_frames):
    """Step the environment and generate multiple noop frames."""
    frames, rewards, dones = [], [], []
    frame, reward, done, _ = env.step(action)                 # take action in environment
    frame = preprocess_frame(frame)
    frames.append(frame)
    rewards.append(reward)
    dones.append(done)
    # 0 is NOOP so the agent doesn't get to move during these frames
    for i in range(n_frames-1):
        frame, reward, done, _ = env.step(0)
        frame = preprocess_frame(frame)
        frames.append(frame)
    state = np.expand_dims(np.asarray(frames), 0)
    reward = sum(rewards)
    done = any(dones)
    return state, reward, done

def remap_action(action, action_map):
    """
    Typically only need to use a subset of the available actions in an environment.
    This defines the mapping that convert actions from a model to desired actions in the environment.
    """
    return action_map[action]
