"""
Training and evaluation runners.
"""

import argparse
import torch
from functions import create_env, create_model, discount, normalize, print_results
from agents import Agent


def train(n_episodes=1000, max_t=1000, gamma=0.99):
    """Training loop."""
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


def evaluate(n_episodes=10, max_t=1000, render=True):
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


# main
parser = argparse.ArgumentParser()
parser.add_argument('--env', help='environment name', type=str, default='CartPole-v0')
parser.add_argument('--eval', help='evaluate (instead of train)', action='store_true')
args = parser.parse_args()

env_name = args.env
print(f'Environment: {env_name}')

if args.eval:
    evaluate()
else:
    train()
