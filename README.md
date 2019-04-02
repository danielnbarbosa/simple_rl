### Introduction
There are many good deep reinforcement algorithm implementations out there but most use Tensorflow.  For PyTorch all I could find were ones based on PyTorch 0.4.  I wanted something that used modern PyTorch semantics and leveraged the library as much as possible.  So this is what I came up with.

As an overarching goal I am trying to keep things as simple as possible and adhere to the original research papers.  Working through this has been very helpful for me to really understand the algorithms.  Feel free to use it as a working baseline and extend it to suit your needs.  If you find issues please let me know!


### Environments
To keep things simple, only OpenAI gym environments are used.  CartPole-v0 is the default because it converges quickly.  Other environments that have been tested include: Acrobot-v1 and LunarLander-v2.


### Algorithms
The following algorithms have been implemented:
- dqn
- reinforce
- reinforce_multi (REINFORCE with multiple parallel environments)
- ppo
- ppo_multi (PPO with multiple parallel environments)


### Dependencies
- python 3.6
- pytorch 1.0
- gym


### Quick Start
First create a virtual environment with all the dependencies installed:
```
conda create -y -n simple_rl python=3.6 anaconda
conda activate simple_rl
conda install -y pytorch torchvision -c pytorch
pip install gym box2d-py
```

Then just select the algorithm you want to use and start training: `./run.py ppo`

When training finishes you can evaluate the model: `./run.py ppo --eval`

You can also run on other environments: `./run.py ppo --env Acrobot-v1`


### Acknowledgements
Thank you to the following code examples that helped me build this.

- https://github.com/udacity/deep-reinforcement-learning
- https://github.com/higgsfield/RL-Adventure/
- https://github.com/higgsfield/RL-Adventure-2/
- https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
