### Summary
REINFORCE is the original policy gradient RL algorithm.  Instead of parameterizing an action-value function it directly parameterizes the policy.  For more details see the original [paper](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf).  It has the following characteristics:
- Policy based
- Model free
- On policy
- Monte Carlo method
- Continuous state space
- Discrete action space


### Model
The model is a straightforward multi-layer perceptron with RELU activations on the hidden layers.  It uses two hidden layers, which is just enough to be considered deep and adequate for low dimensional state spaces.  The final activation is a softmax which gives the probability distribution of the actions.


### Agent
There are two different agents implemented, one for a single environment and one for multiple parallel environments.

The act() method takes a state from the environment and runs a forward pass on the model.  The output of the model is the predicted probability of each action for the given state.  Then the REINFORCE trick is used, which uses a sample trajectory to calculate an estimate of the gradient.  An action is sampled based on the probability distribution, and returned along with the corresponding log probability.

The learn() method uses all the rewards and log probabilities from the trajectory to calculate the gradient.  Gradient ascent is used to move in the direction of greater return which has the effect of doing more good actions and less bad ones.



### Training loop
The outer loop is over episodes, the inner loop over steps.  At each step the agent chooses an action and takes an action according to its current policy.  As REINFORCE is a Monte Carlo method, learning only happens at the end of an episode and uses the normalized, discounted rewards.  Discounting is the standard way to give less weight to rewards that are further into the future, as they are less certain.  Normalization has the effect of encouraging half the actions and discouraging the other half and is helpful for reducing variance.


### Evaluation loop
The evaluation loop is nearly identical to the training loop but doesn't implement learning at the end of the episode.  As actions are sampled from the output of the model, the policy is already stochastic.
