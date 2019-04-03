"""
Training and evaluation runners for learning from pixels in Atari games.

Identical to lowdim runner apart from the following:
- slighty different hyperparameters due to increased training time
- loads CNN model instead of MLP
- does env.reset() with multiple frames
- does env.step() with multiple frames
- remaps actions from output of model to environment
- outputs results/saves model every episode
"""

import time
from collections import namedtuple
import torch
from common.functions import create_env
from common.atari import env_reset_frames, env_step_frames
from .functions import create_cnn, print_results
from .agents import Agent


def train(env_name,
          n_episodes=10000,
          max_t=350,
          gamma=0.99,
          eps_start=1.0,
          eps_end=0.1,
          eps_decay=0.999,
          action_map={0: 4, 1: 5}):
    """Training loop."""
    env = create_env(env_name, max_t)
    models = create_cnn(action_size=len(action_map))
    agent = Agent(models)

    result = namedtuple("Result", field_names=['episode_return', 'epsilon', 'buffer_len', 'steps'])
    results = []
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env_reset_frames(env)

        for t in range(1, max_t+1):
            action = agent.act(state, eps)                                       # select an action
            next_state, reward, done = env_step_frames(env, action_map[action])  # take action in environment
            experience = (state, action, reward, next_state, done)               # build experience tuple
            agent.learn(experience, gamma)                                       # learn from experience
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


def evaluate(env_name, n_episodes=10, max_t=5000, eps=0.05, render=True, action_map={0: 4, 1: 5}):
    """Evaluation loop."""
    env = create_env(env_name, max_t)
    q_net, target_net = create_cnn(action_size=len(action_map))
    q_net.load_state_dict(torch.load('model.pth'))
    agent = Agent((q_net, target_net))

    result = namedtuple("Result", field_names=['episode_return', 'epsilon', 'buffer_len', 'steps'])
    results = []
    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env_reset_frames(env)

        for t in range(1, max_t+1):
            if render:
                #time.sleep(.05)
                env.render()
            action = agent.act(state, eps)                                 # select an action
            state, reward, done = env_step_frames(env, action_map[action]) # take action in environment
            episode_return += reward
            if done:
                r = result(episode_return, eps, 0, t)
                results.append(r)
                break

        print_results(results)
    env.close()
