"""
Agent: works with a single environment.
VectorizedAgent: works with multiple parallel environments.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from common.functions import get_device

device = get_device()


class Agent():
    def __init__(self, model, lr=1e-2):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model = model

    def act(self, state):
        """Given a state, determine the next action by sampling from the action probabilities."""
        # convert ndarray to tensor
        state = torch.from_numpy(state).float().to(device)
        # if state is 1D then expand dim 0 for batch size of 1
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # calculate action probabilities
        probs = self.model.forward(state).cpu().detach()
        # select an action by sampling from probability distribution
        m = Categorical(probs)  # dim = 2
        action = m.sample()     # dim = 1
        # need to squeeze() because env expects scalar for single environment and 1D array for parallel environments
        # need to unsqueeze() because index tensor for gather must have same dimensions as input
        return action.detach().numpy().squeeze(), probs.gather(1, action.unsqueeze(1)).numpy()

    def learn(self, rewards, probs, states, actions, eps):
        """Update model weights."""
        # convert everything to tensors
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # need to squeeze() because probs comes as list of 2D arrays, which would turn into 3D tensor
        probs = torch.tensor(probs, dtype=torch.float, device=device).squeeze()
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        # get new probabilities for actions using current policy
        new_probs = self._calc_new_probs(states, actions)
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

    def _calc_new_probs(self, states, actions):
        """
        Takes states and actions from an old rollout.
        Runs states through the model and returns new probabilities for the action.
        In this case we will use the gradients during backprop so we don't want to detach as in act().
        """
        states = np.asarray(states)
        states = torch.from_numpy(states).float().to(device)
        probs = self.model.forward(states)  # dim = 2
        # need to unsqueeze() because index tensor for gather must have same dimensions as input
        actions = actions.unsqueeze(1)   # dim = 2 (after unsqueeze())
        # need to squeeze() so dimensions line up with old probs
        return probs.gather(1, actions).squeeze(1)
