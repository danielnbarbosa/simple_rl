"""
Training and evaluation runners.
Support multiple parallel environments using OpenAI baselines vectorized environment.
"""

import gym
import torch
import numpy as np
from models import TwoLayerMLP
from functions import make_env, flatten_rollouts, normalize, print_results, get_device
from agents import Agent, VectorizedAgent
from multiprocessing_env import SubprocVecEnv


# create multiple environments for training
env_name = 'CartPole-v0'
num_envs = 4
max_episode_steps = 1000
envs = [make_env(env_name, max_episode_steps) for i in range(num_envs)]
envs = SubprocVecEnv(envs)
print(f'Parallel Environments: {envs.num_envs}')
# create single environment for evaluation
env = gym.make(env_name)
env._max_episode_steps = 1000
# define size of model layers
state_size = envs.observation_space.shape[0]
action_size = envs.action_space.n
hidden_size = (16, 16)
# create model
device = get_device()
model = TwoLayerMLP((state_size, *hidden_size, action_size)).to(device)


def train(n_episodes=100, max_t=1000, gamma=0.99):
    """Training loop."""
    # create agents
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



def eval(n_episodes=1, max_t=1000, render=True):
    """Evaluation loop."""
    # create agent from saved model
    model.load_state_dict(torch.load('model.pth'))
    agent = Agent(model)

    returns = []
    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = env.reset()

        for t in range(1, max_t+1):
            if render: env.render()
            action, _ = agent.act(state)                    # select an action
            state, reward, done, _ = env.step(action)       # take action in environment
            rewards.append(reward)
            if done:
                returns.append(sum(rewards))
                break

        print_results(returns)


# main
train()
envs.close()
eval()
env.close()
