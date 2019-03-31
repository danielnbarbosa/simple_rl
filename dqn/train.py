"""
Training and evaluation runners.
"""

from collections import namedtuple
import gym
import torch
import numpy as np
from functions import print_results, get_device
from agents import Agent
from models import TwoLayerMLP

# create environment
env_name = 'CartPole-v0'
env = gym.make(env_name)
env._max_episode_steps = 1000
# define size of model layers
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = (128, 128)
# create models
device = get_device()
q_net = TwoLayerMLP((state_size, *hidden_size, action_size)).to(device)
target_net = TwoLayerMLP((state_size, *hidden_size, action_size)).to(device)


def train(n_episodes=100, max_t=1000, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
    """Training loop."""
    # create agent
    agent = Agent((q_net, target_net))

    result = namedtuple("Result", field_names=["episode_return", "epsilon", "buffer_len"])
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
                r = result(episode_return, eps, len(agent.memory))
                results.append(r)
                break

        eps = max(eps_end, eps_decay*eps)  # decrease epsilon

        if i_episode % 20 == 0:
            torch.save(agent.q_net.state_dict(), 'model.pth')
            print_results(results)


def eval(n_episodes=1, max_t=1000, eps=0.05, render=True):
    """Evaluation loop."""
    # create agent from saved model
    q_net.load_state_dict(torch.load('model.pth'))
    agent = Agent((q_net, target_net))

    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env.reset()

        for t in range(1, max_t+1):
            if render: env.render()
            action = agent.act(state, eps)                   # select an action
            state, reward, done, _ = env.step(action)   # take action in environment
            episode_return += reward
            if done:
                break

        print(f'Episode: {i_episode}   Reward: {episode_return}')


# main
train()
eval()
env.close()
