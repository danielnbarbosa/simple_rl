"""
Training and evaluation runners for low dimensional state spaces.
"""

from collections import namedtuple
import torch
import numpy as np
from common.functions import create_env
from .functions import create_mlp_models, print_results
from .agents import Agent


def train(env_name, n_episodes=1000, max_t=1000, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
    """Training loop."""
    env = create_env(env_name, max_t)
    models = create_mlp_models(env)
    agent = Agent(models)

    result = namedtuple("Result", field_names=['episode_return', 'epsilon', 'buffer_len', 'steps'])
    results = []
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env.reset()

        for t in range(1, max_t+1):
            action = agent.act(state, eps)                          # select an action
            next_state, reward, done, _ = env.step(action)          # take action in environment
            experience = (state, action, reward, next_state, done)  # build experience tuple
            agent.learn(experience, gamma)                          # learn from experience
            state = next_state
            episode_return += reward
            if done:
                r = result(episode_return, eps, len(agent.memory), t)
                results.append(r)
                break

        eps = max(eps_end, eps_decay*eps)  # decrease epsilon

        if i_episode % 20 == 0:
            torch.save(agent.q_net.state_dict(), 'model.pth')
            print_results(results)
    env.close()


def evaluate(env_name, n_episodes=10, max_t=1000, eps=0.05, render=True):
    """Evaluation loop."""
    env = create_env(env_name, max_t)
    q_net, target_net = create_mlp_models(env)
    q_net.load_state_dict(torch.load('model.pth'))
    agent = Agent((q_net, target_net))

    result = namedtuple("Result", field_names=['episode_return', 'epsilon', 'buffer_len', 'steps'])
    results = []
    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env.reset()

        for t in range(1, max_t+1):
            if render:
                env.render()
            action = agent.act(state, eps)              # select an action
            state, reward, done, _ = env.step(action)   # take action in environment
            episode_return += reward
            if done:
                r = result(episode_return, eps, 0, t)
                results.append(r)
                break

        print_results(results)
    env.close()
