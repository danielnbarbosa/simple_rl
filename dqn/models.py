"""
As models are predicting Q-values, they don't use activation functions.
"""

import torch.nn as nn


class Flatten(nn.Module):
    """Flatten into 2D tensor with first dimension as batch size."""
    def forward(self, x):
        return x.view(x.size()[0], -1)


class MLP(nn.Module):
    """
    Multi layer perceptron with two hidden layers.
    Input shape: [batch_size, inputs].
    Output shape: [batch_size, outputs].
    """
    def __init__(self, inputs, outputs):
        super(MLP, self).__init__()
        fc1, fc2 = (128, 128)
        self.layers = nn.Sequential(
            nn.Linear(inputs, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, outputs)
        )
    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network used in DQN paper.
    Input shape: [batch_size, frames, 84, 84].
    Output shape: [batch_size, outputs].
    """
    def __init__(self, frames, outputs):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(frames, 32, kernel_size=8, stride=4),     #[-1, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),         #[-1, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),         #[-1, 64, 7, 7]
            nn.ReLU(),
            Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, outputs)
        )
    def forward(self, x):
        x = x.float() / 255         # normalize input
        return self.layers(x)
