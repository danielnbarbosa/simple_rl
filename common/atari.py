"""
Functions for processing atari games.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

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


##########          single environment functions          ##########
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



##########          multi environment functions          ##########
def preprocess_frame_multi(frame_arr):
    """Pre-process a single Atari game frame for multiple environments."""
    frames = []
    n_envs = frame_arr.shape[0]
    for n in range(n_envs):
        frame = frame_arr[n]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert to gray scale
        frame = cv2.resize(frame, (84, 84))             # squish
        frames.append(frame)
    return np.asarray(frames)


def generate_frames_multi(envs, n_frames):
    """
    Take actions to generate multiple frames and gather results.
    Actions seem to accumulate in frameskip versions.
    Deterministic-v4 variants have builtin frameskip so use action 0 (NOOP).
    NoFrameskip-v4 variants have no frameskip so use action repeat.
    """

    frames, rewards, dones = [], [], []
    for i in range(n_frames):
        frame, reward, done, _ = envs.step(np.asarray([0] * envs.num_envs))
        frame = preprocess_frame_multi(frame)
        frames.append(frame)
        rewards.append(reward)
        dones.append(done)
    return frames, rewards, dones


def envs_reset_frames(envs, n_frames=4):
    """Reset environment and generate multiple noop frames."""
    frame = envs.reset()                             # reset environment
    frame = preprocess_frame_multi(frame)
    frames, _, _ = generate_frames_multi(envs, n_frames-1) # generate additional frames
    frames.insert(0, frame)
    state = np.asarray(frames, dtype=np.uint8)      # convert to uint8 to save memory
    state = np.swapaxes(state, 0, 1)
    return state


def envs_step_frames(envs, action, action_map, n_frames=4):
    """Step the environment and generate multiple noop frames."""
    action = [action_map[a] for a in action]
    frame, reward, done, _ = envs.step(action)                   # take action in environment
    frame = preprocess_frame_multi(frame)
    frames, rewards, dones = generate_frames_multi(envs, n_frames-1)   # generate additional frames
    frames.insert(0, frame)
    rewards.insert(0, reward)
    dones.insert(0, done)
    state = np.asarray(frames, dtype=np.uint8)                  # convert to uint8 to save memory
    state = np.swapaxes(state, 0, 1)
    reward = np.sum(np.asarray(rewards), axis=0)
    done = np.any(np.asarray(dones), axis=0)
    return state, reward, done
