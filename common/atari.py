"""
Shared Atari related functions.
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
