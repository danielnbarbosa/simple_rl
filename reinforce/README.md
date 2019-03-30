### Summary
REINFORCE is the original policy based RL algorithm.  Instead of parameterizing an action-value function it directly parameterizes the policy.  It has the following characteristics:
- Policy based
- Model free
- On policy
- Monte Carlo method
- Continuous state space
- Discrete action space
For more details see the original [paper](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)


### Model
This is a straightforward multi-layer perceptron with RELU activations on the hidden layers.  It uses two layers, which is just enough to be considered deep, and adequate for low dimensional state spaces.  The final activation is a softmax which gives us the probability distribution of the actions.


### Agent


### Training loop


### Evaluation loop
