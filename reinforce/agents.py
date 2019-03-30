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
        """Given a state, determine the next action."""
        # convert ndarray to tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # calculate action probabilities
        probs = self.model.forward(state).cpu()
        # select an action by sampling from probability distribution
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def learn(self, rewards, log_probs):
        """Update model weights."""
        losses = []
        for i, log_prob in enumerate(log_probs):
            losses.append(-log_prob * rewards[i])

        loss = torch.cat(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class VectorizedAgent():
    def __init__(self, model, lr=1e-2):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model = model

    def act(self, state):
        """Given a state, determine the next action."""
        # convert ndarray to tensor, has num_envs as first dimension
        state = torch.from_numpy(state).float().to(device)
        # calculate action probabilities
        probs = self.model.forward(state).cpu()
        # select an action by sampling from probability distribution
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)

    def learn(self, rewards, log_probs):
        """Update model weights."""
        losses = []
        for i, log_prob in enumerate(log_probs):
            losses.append((-log_prob * rewards[i]).unsqueeze(0))

        loss = torch.cat(losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
