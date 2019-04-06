"""
Misc functions.
"""

import torch
import numpy as np


def get_device():
    """Check if GPU is is_available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def moving_average(values, window=100):
    """Calculate moving average over window."""
    # don't run until collecting at least window results
    if len(values) < window:
        return [-np.inf]
    else:
        weights = np.repeat(1.0, window)/window
        return np.convolve(values, weights, 'valid')
