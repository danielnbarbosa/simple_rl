"""
Agent: works with a single environment.
VectorizedAgent: works with multiple parallel environments.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from functions import get_device

device = get_device()


class Agent():
    def __init__(self, model, lr=1e-2):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model = model

    def act(self, state):
        """Given a state, determine the next action by sampling from the action probabilities."""
        # convert ndarray to tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # calculate action probabilities
        probs = self.model.forward(state).cpu().detach()
        # select an action by sampling from probability distribution
        m = Categorical(probs)
        action = m.sample()
        return action.item(), probs.gather(1, action.unsqueeze(1)).numpy()

    def learn(self, rewards, probs, states, actions, eps):
        """Update model weights."""
        # convert everything to tensors
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        probs = torch.tensor(probs, dtype=torch.float, device=device).squeeze()
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        # get new probabilities for actions using current policy
        new_probs = self._calc_probs(states, actions)
        # ratio for clipping
        ratio = new_probs/probs
        # clipped function
        clip = torch.clamp(ratio, 1-eps, 1+eps)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
        # backprop
        loss = -torch.sum(clipped_surrogate)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _calc_probs(self, states, actions):
        """
        Given states and actions, run states through the model.
        Return new probabilities for the action.
        """
        states = np.asarray(states)
        states = torch.from_numpy(states).float().to(device)
        probs = self.model.forward(states)
        actions = actions.unsqueeze(1)
        return probs.gather(1, actions).squeeze(1)


class VectorizedAgent():
    def __init__(self, model, lr=1e-2):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model = model

    def act(self, state):
        """Given a state, determine the next action."""
        # convert ndarray to tensor, has num_envs as first dimension
        state = torch.from_numpy(state).float().to(device)
        # calculate action probabilities
        probs = self.model.forward(state).cpu().detach()
        # select an action by sampling from probability distribution
        m = Categorical(probs)
        action = m.sample()
        return action.detach().numpy(), probs.gather(1, action.unsqueeze(1)).numpy()

    def learn(self, rewards, probs, states, actions, eps):
        """Update model weights."""
        # convert everything to tensors
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        probs = torch.tensor(probs, dtype=torch.float, device=device).squeeze()
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        # get new probabilities for actions using current policy
        new_probs = self._calc_probs(states, actions)
        # ratio for clipping
        ratio = new_probs/probs
        # clipped function
        clip = torch.clamp(ratio, 1-eps, 1+eps)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
        # backprop
        loss = -torch.sum(clipped_surrogate)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _calc_probs(self, states, actions):
        """
        Given states and actions, run states through the model.
        Return new probabilities for the action.
        """
        states = torch.from_numpy(states).float().to(device)
        probs = self.model.forward(states)
        actions = actions.unsqueeze(1)
        return probs.gather(1, actions).squeeze(1)
