"""
Agent: works with a single environment.
VectorizedAgent: works with multiple parallel environments.
"""

import torch
import torch.optim as optim
from torch.distributions import Categorical


class Agent():
    def __init__(self, device, model, lr=1e-2):
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model = model

    def act(self, state):
        """Given a state, determine the next action."""
        # convert ndarray to tensor
        state = torch.from_numpy(state).float().to(self.device)
        # if state is 1D then expand dim 0 for batch size of 1
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # calculate action probabilities
        probs = self.model.forward(state).cpu() # dim = 2
        # select an action by sampling from probability distribution
        m = Categorical(probs)
        action = m.sample()  # dim = 1
        # need to squeeze because env expects scalar for single environment and 1D array for parallel environments
        return action.detach().squeeze().numpy(), m.log_prob(action)

    def learn(self, rewards, log_probs):
        """Update model weights."""
        losses = []
        for i, log_prob in enumerate(log_probs):
            losses.append(-log_prob * rewards[i])
        loss = torch.cat(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
