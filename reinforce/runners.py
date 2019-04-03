"""
Training and evaluation runners.
Supports multiple parallel environments using OpenAI baselines vectorized environment.
"""

from collections import namedtuple
import torch
import numpy as np
from common.functions import create_env, create_envs, discount, normalize
from .functions import create_mlp, flatten_rollouts, print_results
from .agents import Agent


def train(env_name,
          n_episodes=1000,
          max_t=1000,
          gamma=0.99):
    """Training loop for a single environment."""
    env = create_env(env_name, max_t)
    model = create_mlp(env)
    agent = Agent(model)
    result = namedtuple("Result", field_names=['episode_return', 'steps'])
    results = []

    for i_episode in range(1, n_episodes+1):
        rewards, log_probs = [], []
        state = env.reset()

        # generate rollout
        for t in range(1, max_t+1):
            action, log_prob = agent.act(state)        # select an action
            state, reward, done, _ = env.step(action)  # take action in environment
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break

        # update model weights
        processed_rewards = normalize(discount(rewards, gamma))
        agent.learn(processed_rewards, log_probs)
        # gather results
        r = result(sum(rewards), t)
        results.append(r)
        if i_episode % 20 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(results)
    env.close()


def train_multi(env_name,
                n_episodes=1000,
                max_t=1000,
                gamma=0.99,
                num_envs=4):
    """Training loop for multiple parallel environments."""
    envs = create_envs(env_name, max_t, num_envs)
    model = create_mlp(envs)
    agent = Agent(model)
    result = namedtuple("Result", field_names=['episode_return', 'steps'])
    results = []

    for i_episode in range(1, n_episodes+1):
        episode_done = [False] * num_envs            # sticky done, as done flag from environment does not persist across steps
        rollouts = {n: [] for n in range(num_envs)}  # rollouts are a dict indexed by agent id
        state = envs.reset()

        # generate rollouts for parallel agents
        for t in range(1, max_t+1):
            action, log_prob = agent.act(state)
            state, reward, done, _ = envs.step(action)
            # separate results by agent
            for n in range(num_envs):
                if episode_done[n] is False:
                    # need to expand 0dim because it gets eaten by dict
                    rollouts[n].append((reward[n], log_prob[n].unsqueeze(0)))
                if done[n]:
                    episode_done[n] = True
            if all(episode_done):
                break

        # flatten rollouts
        rewards, discounted_rewards, log_probs = flatten_rollouts(rollouts, gamma)
        # update model weights
        processed_rewards = normalize(discounted_rewards)  # flatten already discounts rewards
        agent.learn(processed_rewards, log_probs)
        # gather results
        r = result(np.sum(rewards)/num_envs, t/num_envs)  # use raw rewards averaged over number of rollouts
        results.append(r)
        if i_episode % 20 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(results)
    envs.close()


def evaluate(env_name, n_episodes=10, max_t=1000, render=True):
    """Evaluation loop."""
    env = create_env(env_name, max_t)
    model = create_mlp(env)
    model.load_state_dict(torch.load('model.pth'))
    agent = Agent(model)
    result = namedtuple("Result", field_names=['episode_return', 'steps'])
    results = []

    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = env.reset()

        for t in range(1, max_t+1):
            if render:
                env.render()
            action, _ = agent.act(state)                    # select an action
            state, reward, done, _ = env.step(action)       # take action in environment
            rewards.append(reward)
            if done:
                break

        # gather results
        r = result(sum(rewards), t)
        results.append(r)
        print_results(results)
    env.close()
