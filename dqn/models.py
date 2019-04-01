"""
Models.
"""

import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """ MLP with two hidden layers."""
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
