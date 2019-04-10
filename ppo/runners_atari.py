"""
Training and evaluation runners.
Supports multiple parallel environments using OpenAI baselines vectorized environment.
"""

from collections import namedtuple
import torch
import numpy as np
from common.misc import get_device
from common.environments import create_env, create_envs
from common.atari import env_reset_frames, env_step_frames, envs_reset_frames, envs_step_frames
from common.rollouts import discount, normalize, discount_and_flatten_rewards, flatten
from .functions import create_cnn, print_results, unzip_rollouts
from .agents import Agent


def train(env_name,
         n_episodes=100000,
         max_t=400,
         gamma=0.999,
         eps=0.1,
         eps_decay=0.999,
         n_updates=4,
         action_map={0: 4, 1: 5}):
    """Training loop for a single environment."""
    device = get_device()
    env = create_env(env_name, max_t)
    model = create_cnn(device, action_size=len(action_map))
    agent = Agent(device, model, lr=1e-4)
    result = namedtuple("Result", field_names=['episode_return', 'epslions', 'steps'])
    results = []

    for i_episode in range(1, n_episodes+1):
        rewards, probs, states, actions = [], [], [], []
        state = env_reset_frames(env)

        # generate rollout
        for t in range(1, max_t+1):
            action, prob = agent.act(state)        # select an action
            action = action.item()
            next_state, reward, done = env_step_frames(env, action_map[action])  # take action in environment
            probs.append(prob)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            if done:
                break

        # process rewards
        normalized_rewards = normalize(discount(rewards, gamma))
        # update model weights
        for _ in range(n_updates):
            agent.learn(normalized_rewards, probs, states, actions, eps)
        # gather results
        r = result(sum(rewards), eps, t)
        results.append(r)
        if i_episode % 10 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(results)
        # decay epsilon
        eps *= eps_decay
    env.close()


def train_multi(env_name,
                n_episodes=100000,
                max_t=400,
                gamma=0.999,
                num_envs=12,
                eps=0.1,
                eps_decay=0.999,
                n_updates=4,
                action_map={0: 4, 1: 5}):
    """Training loop for multiple parallel environments."""
    device = get_device()
    envs = create_envs(env_name, max_t, num_envs)
    model = create_cnn(device, action_size=len(action_map))
    agent = Agent(device, model, lr=1e-4)
    result = namedtuple("Result", field_names=['episode_return', 'epslions', 'steps'])
    results = []

    for i_episode in range(1, n_episodes+1):
        episode_done = [False] * num_envs            # sticky done, as done flag from environment does not persist across steps
        rollouts = {n: [] for n in range(num_envs)}  # rollouts are a dict indexed by environment id
        state = envs_reset_frames(envs)

        # generate rollouts for parallel agents
        for t in range(1, max_t+1):
            action, prob = agent.act(state)
            next_state, reward, done = envs_step_frames(envs, action, action_map)
            # separate results by agent
            for n in range(num_envs):
                if episode_done[n] is False:
                    # append results to the list of the associated environment id
                    rollouts[n].append((reward[n], prob[n], state[n], action[n]))
                if done[n]:
                    episode_done[n] = True
            state = next_state
            if all(episode_done):
                break

        rewards, probs, states, actions = unzip_rollouts(rollouts)
        # process rewards
        discounted_rewards = discount_and_flatten_rewards(rewards, gamma)
        normalized_rewards = normalize(discounted_rewards)
        # flatten rollouts
        rewards = flatten(rewards)
        probs = flatten(probs)
        states = flatten(states)
        actions = flatten(actions)
        # update model weights
        for _ in range(n_updates):
            agent.learn(normalized_rewards, probs, states, actions, eps)
        # gather results
        r = result(np.sum(rewards)/num_envs, eps, t)  # use raw rewards averaged over number of rollouts, all steps
        results.append(r)
        if i_episode % 10 == 0:
            torch.save(agent.model.state_dict(), 'model.pth')
            print_results(results)
        # decay epsilon
        eps *= eps_decay
    envs.close()


def evaluate(env_name, n_episodes=10, max_t=5000, render=True, action_map={0: 4, 1: 5}):
    """Evaluation loop."""
    device = get_device()
    env = create_env(env_name, max_t)
    model = create_cnn(device, action_size=len(action_map))
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    agent = Agent(device, model)
    result = namedtuple("Result", field_names=['episode_return', 'epslions', 'steps'])
    results = []

    model.eval()
    with torch.no_grad():
        for i_episode in range(1, n_episodes+1):
            rewards = []
            state = env_reset_frames(env)

            for t in range(1, max_t+1):
                if render:
                    env.render()
                action, _ = agent.act(state)                    # select an action
                action = action.item()
                state, reward, done = env_step_frames(env, action_map[action])       # take action in environment
                rewards.append(reward)
                if done:
                    break

            # gather results
            r = result(sum(rewards), 0, t)
            results.append(r)
            print_results(results)
        env.close()
