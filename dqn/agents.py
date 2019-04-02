"""
DQN Agent.
"""

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .memory import ReplayBuffer
from .functions import get_device


device = get_device()

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, models,
                 buffer_size=int(1e5),
                 batch_size=64,
                 update_freq=int(1e3),
                 lr=5e-4):

        self.batch_size = batch_size
        self.update_freq = update_freq
        self.model_updates = 0

        # initialize q_net and target_net with same initial values
        self.q_net, self.target_net = models
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, self.batch_size)


    def act(self, state, eps=0.):
        """Given a state, determine the next action."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # calculate action values
        self.q_net.eval()
        with torch.no_grad():
            action_values = self.q_net(state)
        self.q_net.train()

        # epsilon-greedy action selection
        action_size = len(action_values.squeeze())
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(action_size))


    def learn(self, experience, gamma):
        """Learn by sampling minibatches of experience from replay buffer."""
        # add latest experience to memory
        self.memory.add(*experience)

        # if enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self._backprop_loss(experiences, gamma)
            self.model_updates += 1
            # replace target_net parameters with q_net ones every update_freq parameter updates
            if self.model_updates % self.update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())


    def _backprop_loss(self, experiences, gamma):
        """Update q network parameters using mini-batches of experience."""
        states, actions, rewards, next_states, dones = experiences
        # get q values for chosen actions
        q_expected = self.q_net(states).gather(1, actions)
        # get max q values for next_states using the fixed network
        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # calculate q values for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
