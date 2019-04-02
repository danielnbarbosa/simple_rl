"""
Training and evaluation runners.
"""

import torch
from .functions import create_env, create_model, discount, normalize, print_results
from .agents import Agent


def train(env_name, n_episodes=1000, max_t=1000, gamma=0.99, eps=0.2, eps_decay=0.999, n_updates=4):
    """Training loop."""
    env = create_env(env_name, max_t)
    model = create_model(env)
    agent = Agent(model)

    returns, epsilons = [], []
    for i_episode in range(1, n_episodes+1):
        rewards, probs, states, actions = [], [], [], []
        state = env.reset()

        for t in range(1, max_t+1):
            action, prob = agent.act(state)        # select an action
            next_state, reward, done, _ = env.step(action)  # take action in environment
            probs.append(prob)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            if done:
                returns.append(sum(rewards))
                epsilons.append(eps)
                break
        # after episode is over
        rewards = normalize(discount(rewards, gamma))  # normalize discounted rewards
        for _ in range(n_updates):
            agent.learn(rewards, probs, states, actions, eps)    # update model weights
        eps *= eps_decay

        if i_episode % 20 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(returns, epsilons)
    env.close()


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
