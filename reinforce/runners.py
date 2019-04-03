"""
Training and evaluation runners.
Supports multiple parallel environments using OpenAI baselines vectorized environment.
"""

import torch
import numpy as np
from common.functions import create_env, make_env, create_envs, discount, normalize
from .functions import create_model, flatten_rollouts, print_results
from .agents import Agent, VectorizedAgent


def train(env_name, n_episodes=1000, max_t=1000, gamma=0.99):
    """Training loop for a single environment."""
    env = create_env(env_name, max_t)
    model = create_model(env)
    agent = Agent(model)

    returns = []
    for i_episode in range(1, n_episodes+1):
        rewards, log_probs = [], []
        state = env.reset()

        for t in range(1, max_t+1):
            action, log_prob = agent.act(state)        # select an action
            state, reward, done, _ = env.step(action)  # take action in environment
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                returns.append(sum(rewards))
                break

        rewards = normalize(discount(rewards, gamma))  # normalize discounted rewards
        agent.learn(rewards, log_probs)                # update model weights

        if i_episode % 20 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(returns)
    env.close()


def train_multi(env_name, n_episodes=1000, max_t=1000, gamma=0.99, num_envs=4):
    """Training loop for multiple parallel environments."""
    envs = create_envs(env_name, max_t, num_envs)
    model = create_model(envs)
    agent = VectorizedAgent(model)

    returns = []
    for i_episode in range(1, n_episodes+1):
        # sticky done, as done flag from environment does not persist across steps
        episode_done = [False] * num_envs
        # rollouts are a dict indexed by agent id
        rollouts = {n: [] for n in range(num_envs)}
        state = envs.reset()

        # generate rollout for each agent
        for t in range(1, max_t+1):
            action, log_prob = agent.act(state)
            state, reward, done, _ = envs.step(action.detach().numpy())

            for n in range(num_envs):
                if episode_done[n] is False:
                    rollouts[n].append((reward[n], log_prob[n]))
                if done[n]:
                    episode_done[n] = True

            if all(episode_done):
                break

        # flatten rollouts into one dimension
        rewards, discounted_rewards, log_probs = flatten_rollouts(rollouts, gamma)
        # backprop gradient across all rollouts using normalized rewards
        agent.learn(normalize(discounted_rewards), log_probs)
        # use raw rewards to calcuate return per episode averaged across number of rollouts
        returns.append(np.sum(rewards) / num_envs)
        if i_episode % 20 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(returns)
    envs.close()


def evaluate(env_name, n_episodes=10, max_t=1000, render=True):
    """Evaluation loop."""
    env = create_env(env_name, max_t)
    model = create_model(env)
    model.load_state_dict(torch.load('model.pth'))
    agent = Agent(model)

    returns = []
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
                returns.append(sum(rewards))
                break

        print_results(returns)
    env.close()