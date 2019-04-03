"""
Training and evaluation runners.
"""

import time
from collections import namedtuple
import torch
import numpy as np
from common.functions import create_env, remap_action, env_reset_with_frames, env_step_with_frames
from .functions import create_cnn_models, print_results
from .agents import Agent


def train(env_name, n_episodes=10000, max_t=350, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=0.999):
    """Training loop."""
    env = create_env(env_name, max_t)
    #models = create_models(env)
    models = create_cnn_models(frames=4, action_size=2)
    agent = Agent(models)

    result = namedtuple("Result", field_names=['episode_return', 'epsilon', 'buffer_len', 'steps'])
    results = []
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env_reset_with_frames(env, 4)

        for t in range(1, max_t+1):
            action = agent.act(state, eps)                          # select an action
            env_action = remap_action(action, {0: 4, 1: 5})
            next_state, reward, done = env_step_with_frames(env, env_action, 4)  # take action in environment
            experience = (state, action, reward, next_state, done)  # build experience tuple
            agent.learn(experience, gamma)                          # learn from experience
            state = next_state
            episode_return += reward
            if done:
                r = result(episode_return, eps, len(agent.memory), t)
                results.append(r)
                break

        eps = max(eps_end, eps_decay*eps)  # decrease epsilon

        if i_episode % 1 == 0:
            torch.save(agent.q_net.state_dict(), 'model.pth')
            print_results(results)
    env.close()


def evaluate(env_name, n_episodes=10, max_t=5000, eps=0.05, render=True):
    """Evaluation loop."""
    env = create_env(env_name, max_t)
    q_net, target_net = create_cnn_models(frames=4, action_size=2)
    q_net.load_state_dict(torch.load('model.pth'))
    agent = Agent((q_net, target_net))

    result = namedtuple("Result", field_names=['episode_return', 'epsilon', 'buffer_len', 'steps'])
    results = []
    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env_reset_with_frames(env, 4)

        for t in range(1, max_t+1):
            if render:
                time.sleep(.05)
                env.render()
            action = agent.act(state, eps)              # select an action
            env_action = remap_action(action, {0: 4, 1: 5})
            state, reward, done = env_step_with_frames(env, env_action, 4) # take action in environment
            episode_return += reward
            if done:
                r = result(episode_return, eps, 0, t)
                results.append(r)
                break

        print_results(results)
    env.close()
