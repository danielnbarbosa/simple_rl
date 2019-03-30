"""
DQN Agent.
"""

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from memory import ReplayBuffer
from functions import get_device


device = get_device()

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, models,
                 buffer_size=int(1e5),
                 batch_size=64,
                 tau=1e-3,
                 lr=5e-4):

        self.batch_size = batch_size
        self.tau = tau

        # initialize live and fixed q networks with same initial values
        self.live_net, self.fixed_net = models
        self.fixed_net.load_state_dict(self.live_net.state_dict())

        self.optimizer = optim.Adam(self.live_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, self.batch_size)


    def act(self, state, eps=0.):
        """Given a state, determine the next action."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # calculate action values
        self.live_net.eval()
        with torch.no_grad():
            action_values = self.live_net(state)
        self.live_net.train()

        # epsilon-greedy action selection
        action_size = len(action_values.squeeze())
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(action_size))


    def learn(self, experience, gamma):
        """Learn from experience."""
        self.memory.add(*experience)

        # if enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self._backprop_loss(experiences, gamma)
            self._soft_update(self.live_net, self.fixed_net, self.tau)


    def _backprop_loss(self, experiences, gamma):
        """Update live model weights using mini-batches of experience."""
        states, actions, rewards, next_states, dones = experiences
        # get q values for chosen actions
        q_expected = self.live_net(states).gather(1, actions)
        # get max q values for next_states using the fixed network
        q_targets_next = self.fixed_net(next_states).detach().max(1)[0].unsqueeze(1)
        # calculate q values for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def _soft_update(self, live_net, fixed_net, tau):
        """Update fixed model weights."""
        for fixed_param, live_param in zip(fixed_net.parameters(), live_net.parameters()):
            fixed_param.data.copy_(tau*live_param.data + (1.0-tau)*fixed_param.data)
