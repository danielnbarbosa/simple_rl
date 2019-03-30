### Introduction
These implementations of popular deep reinforcement learning algorithms are meant to be as simple as possible.  It should be good for really learning and understanding the essence of the algorithms without getting bogged down in extraneous details or unnecessary abstractions.  Feel free to use it as a working baseline and extend it to suit your needs.

It is a modern implementation that uses PyTorch 1.0 and python 3.6.


### Environments
All examples use the OpenAI gym CartPole-v0 environment because it is a simple one that converges relatively quickly.  It's a good first step when testing any new RL algorithm.


### Algorithms
The following algorithms have been implemented:
- DQN
- REINFORCE
- REINFORCE multiprocessing



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

Then just select the algorithm you want to use and run the train script.
```
cd dqn
python train.py
```



### Acknowledgements
Thank you to the following code examples that helped me build this.

- https://github.com/udacity/deep-reinforcement-learning
- https://github.com/higgsfield/RL-Adventure/
- https://github.com/higgsfield/RL-Adventure-2/
- https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
