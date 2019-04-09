"""
Run tests using nosetests.
"""

from nose.tools import assert_equals
import gym
import numpy as np
from common.rollouts import discount, normalize, flatten, discount_and_flatten_rewards
from common.atari import env_reset_frames, env_step_frames
from common.misc import moving_average

##########          rollouts          ##########
def test_discount():
    rewards = [10.0, 0.0, -50.0]
    gamma = 0.8
    result = np.array([-22., -40., -50.])
    assert np.array_equal(discount(rewards, gamma), result) == True

def test_normalize():
    rewards = [4.,3.,2.,1.]
    result = np.array([1.34164079,  0.4472136 , -0.4472136 , -1.34164079])
    assert np.allclose(normalize(rewards), result) == True

def test_moving_average():
    values = [4.,3.,2.,1.]
    result = np.array([3.5, 2.5, 1.5])
    assert np.array_equal(moving_average(values, window=2), result) == True

def test_flatten():
    value = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
    result = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert_equals(flatten(value), result)

def test_discount_and_flatten_rewards():
    value = {0: [1.], 1: [1., 1.], 2: [1., 1., 1.]}
    result = np.array([1., 1.99, 1., 2.9701, 1.99, 1.])
    assert np.array_equal(discount_and_flatten_rewards(value, 0.99), result)


##########          atari          ##########
def test_env_reset_frames():
    env = gym.make('PongDeterministic-v4')
    frames = env_reset_frames(env)
    assert_equals(frames.shape, (1, 4, 84, 84))

def test_env_step_frames():
    env = gym.make('PongDeterministic-v4')
    frames = env_reset_frames(env)
    assert_equals(frames.shape, (1, 4, 84, 84))
