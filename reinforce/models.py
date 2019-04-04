"""
As models are predicting action probabilities they use a softmax activation function.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Multi layer perceptron with two hidden layers.
    Input shape: [batch_size, inputs].
    Output shape: [batch_size, outputs].
    """
    def __init__(self, inputs, outputs):
        super(MLP, self).__init__()
        fc1, fc2 = (16, 16)
        self.layers = nn.Sequential(
            nn.Linear(inputs, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, outputs),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layers(x)
