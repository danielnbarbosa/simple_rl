import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """MLP with two hidden layers."""

    def __init__(self, layer_sizes):
        super(TwoLayerMLP, self).__init__()

        inputs, fc1, fc2, outputs = layer_sizes
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
