"""
Training and evaluation runners.
Support multiple parallel environments using OpenAI baselines vectorized environment.
"""

import torch
import numpy as np
from .functions import create_env, create_envs, create_model, flatten_rollouts, normalize, print_results
from .agents import Agent, VectorizedAgent


def train(env_name, n_episodes=1000, max_t=1000, gamma=0.99, num_envs=4, eps=0.2, eps_decay=0.999, n_updates=4):
    """Training loop."""
    envs = create_envs(env_name, max_t, num_envs)
    model = create_model(envs)
    agent = VectorizedAgent(model)

    returns, epsilons = [], []
    for i_episode in range(1, n_episodes+1):
        # sticky done, as done flag from environment does not persist across steps
        episode_done = [False] * num_envs
        # rollouts are a dict indexed by agent id
        rollouts = {n: [] for n in range(num_envs)}
        state = envs.reset()

        # generate rollout for each agent
        for t in range(1, max_t+1):
            action, prob = agent.act(state)
            next_state, reward, done, _ = envs.step(action)

            for n in range(num_envs):
                if episode_done[n] is False:
                    rollouts[n].append((reward[n], prob[n], state[n], action[n]))
                if done[n]:
                    episode_done[n] = True

            state = next_state
            if all(episode_done):
                break

        epsilons.append(eps)
        # flatten rollouts into one dimension
        rewards, discounted_rewards, probs, states, actions = flatten_rollouts(rollouts, gamma)
        # backprop gradient across all rollouts using normalized rewards
        for _ in range(n_updates):
            agent.learn(normalize(discounted_rewards), probs, states, actions, eps)
        eps *= eps_decay

        # use raw rewards to calcuate return per episode averaged across number of rollouts
        returns.append(np.sum(rewards) / num_envs)
        if i_episode % 20 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(returns, epsilons)
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
