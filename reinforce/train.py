"""
Training and evaluation runners.
"""

import gym
import torch
from models import TwoLayerMLP
from functions import discount, normalize, print_results, get_device
from agents import Agent


# create environment
env_name = 'CartPole-v0'
env = gym.make(env_name)
env._max_episode_steps = 1000
# define size of model layers
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = (16, 16)
# create model
device = get_device()
model = TwoLayerMLP((state_size, *hidden_size, action_size)).to(device)


def train(n_episodes=400, max_t=1000, gamma=0.99):
    """Training loop."""
    # create agent
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



def eval(n_episodes=1, max_t=1000, render=True):
    """Evaluation loop."""
    # create agent from saved model
    model.load_state_dict(torch.load('model.pth'))
    agent = Agent(model)

    for i_episode in range(1, n_episodes+1):
        episode_return = 0
        state = env.reset()

        for t in range(1, max_t+1):
            if render: env.render()
            action, _ = agent.act(state)                    # select an action
            state, reward, done, _ = env.step(action)       # take action in environment
            episode_return += reward
            if done:
                break

        print(f'Episode: {i_episode}   Reward: {episode_return}')


# main
#train()
eval()
env.close()
