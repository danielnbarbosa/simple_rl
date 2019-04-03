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
def is_atari(env_name):
    """Determine if environment is an Atari game."""
    games = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']
    return any([env_name.startswith(game.capitalize()) for game in games])

def preprocess_frame(frame):
    """Pre-process a single Atari game frame."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert to gray scale
    frame = cv2.resize(frame, (84, 84))             # squish
    return frame

def generate_frames(env, n_frames):
    """
    Take actions to generate multiple frames and gather results.
    Actions seem to accumulate in frameskip versions.
    Deterministic-v4 variants have builtin frameskip so use action 0 (NOOP).
    NoFrameskip-v4 variants have no frameskip so use action repeat.
    """
    frames, rewards, dones = [], [], []
    for i in range(n_frames):
        frame, reward, done, _ = env.step(0)
        frame = preprocess_frame(frame)
        frames.append(frame)
        rewards.append(reward)
        dones.append(done)
    return frames, rewards, dones

def env_reset_frames(env, n_frames=4):
    """Reset environment and generate multiple noop frames."""
    frame = env.reset()                             # reset environment
    frame = preprocess_frame(frame)
    frames, _, _ = generate_frames(env, n_frames-1) # generate additional frames
    frames.insert(0, frame)
    state = np.asarray(frames, dtype=np.uint8)      # convert to uint8 to save memory
    state = np.expand_dims(state, 0)                # expand dim0 for batch size of 1
    return state

def env_step_frames(env, action, n_frames=4):
    """Step the environment and generate multiple noop frames."""
    frame, reward, done, _ = env.step(action)                   # take action in environment
    frame = preprocess_frame(frame)
    frames, rewards, dones = generate_frames(env, n_frames-1)   # generate additional frames
    frames.insert(0, frame)
    rewards.insert(0, reward)
    dones.insert(0, done)
    state = np.asarray(frames, dtype=np.uint8)                  # convert to uint8 to save memory
    state = np.expand_dims(state, 0)                            # expand dim0 for batch size of 1
    reward = sum(rewards)
    done = any(dones)
    return state, reward, done

def remap_action(action, action_map):
    """
    Typically only need to use a subset of the available actions in an environment.
    This maps actions from a model to desired actions in the environment.
    """
    return action_map[action]
