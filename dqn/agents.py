"""
DQN Agent.
"""

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .memory import ReplayBuffer


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, device, models,
                 buffer_size=int(2e5),
                 batch_size=32,
                 target_net_update=int(1e4),
                 lr=2.5e-4):

        self.device = device
        self.batch_size = batch_size
        self.target_net_update = target_net_update
        self.model_updates = 0

        # initialize q_net and target_net with same initial values
        self.q_net, self.target_net = models
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, self.batch_size)


    def act(self, state, eps=0.):
        """Given a state, determine the next action."""
        # convert ndarray to tensor
        state = torch.from_numpy(state).float().to(self.device)
        # if state is 1D then expand dim 0 for batch size of 1
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # calculate action values
        self.q_net.eval()
        with torch.no_grad():
            action_values = self.q_net(state)
        self.q_net.train()

        # epsilon-greedy action selection
        action_size = action_values.size()[1]
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
            # replace target_net parameters with q_net ones every so often
            if self.model_updates % self.target_net_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())


    def _backprop_loss(self, experiences, gamma):
        """Update q network parameters using mini-batches of experience."""
        states, actions, rewards, next_states, dones = experiences
        # get q values for chosen actions
        q_expected = self.q_net(states).gather(1, actions)
        # get max q values for next_states using the fixed network
        # max() returns a tuple (max value, argmax), we only want the first part
        # unsqueeze(1) to bring back dimension lost by max() operation
        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # calculate q values for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # use mean squared error to calculate loss
        loss = F.mse_loss(q_expected, q_targets)
        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
