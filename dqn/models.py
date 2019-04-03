"""
Models.
"""

import torch.nn as nn
from torchvision.transforms import transforms

class TwoLayerMLP(nn.Module):
    """MLP with two hidden layers."""
    def __init__(self, inputs, outputs):
        super(TwoLayerMLP, self).__init__()
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


class Flatten(nn.Module):
    def forward(self, x):
        # flatten into 2D tensors with first dimension as batch size
        return x.view(x.size()[0], -1)


class ConvNet(nn.Module):
    """CNN from DQN paper."""
    def __init__(self, frames, outputs):
        super(ConvNet, self).__init__()
        #input                                                   [-1, frames, 84, 84]
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
        return self.layers(x)
