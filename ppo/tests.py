"""
Run tests using nosetests.
"""

import numpy as np
from functions import discount, normalize, moving_average

def test_discount():
    rewards = [10.0, 0.0, -50.0]
    gamma = 0.8
    result = np.array([-22., -40., -50.])
    assert np.array_equal(discount(rewards, gamma), result) == True

def test_normalize():
    # didn't confirm accuracy of this
    rewards = [4.,3.,2.,1.]
    result = np.array([1.34164079,  0.4472136 , -0.4472136 , -1.34164079])
    assert np.allclose(normalize(rewards), result) == True

def test_moving_average():
    values = [4.,3.,2.,1.]
    result = np.array([3.5, 2.5, 1.5])
    assert np.array_equal(moving_average(values, window=2), result) == True
