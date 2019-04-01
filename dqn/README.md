### Summary
DQN merges traditional Q-learning with a deep neural network.  It uses value iteration to approximate the optimal action-value function.  For more details see the original [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).  It has the following characteristics:
- Value based
- Model free
- Off policy
- Temporal difference method
- Continuous state space
- Discrete action space


### Model
The model is a straightforward multi-layer perceptron with RELU activations on the hidden layers.  It uses two hidden layers, which is just enough to be considered deep and adequate for low dimensional state spaces.  As it outputs the predicted Q values for all actions there is no final activation function.


### Experience Replay Buffer
The experience replay buffer is a circular buffer of fixed length.  This is one of two important tweaks that help the algorithm converge more reliably.  Instead of training using the immediate interactions with the environment, the agent randomly samples from the buffer.  This helps break any temporal correlations that could cause the model to diverge.

The replay buffer is a deque of named tuples.  Each tuple consists of (state, action, reward, next_state, done).  The experience sampling is done with a fixed batch size and then converted to tensors for feeding to the network for training.  The batch size is just like the minibatch size in supervised learning and training is done in these sized batches as well.


### Agent
During initialization the agent creates two Q networks with identical weights.  This brings us to the second important tweak that helps DQN converge: fixed Q-targets.  Due to the fact the Q learning uses bootstrapping, the TD target can change due to the generalizing effects of the neural network, which can cause the model to get stuck in a self referential feedback loop.  To prevent this, a separate (target) Q network is used with values that are only occasionally synced with the normal Q network.

The first method is act(), which defines the action an agent takes for a given state (and epsilon value).   The agent does a forward pass on the state which deterministically produces predicted action values.  Then it uses an epsilon greedy policy to decide if it should behave greedily or uniformly at random.

The second method is learn().  This is where the meat of the algorithm is.  First it stores the latest experience in memory.  Then it samples a minibatch from the memory and learns from it's stored experience.

First it calculates the expected q values for the current state using the q network.  Then it calculates the max q values for the next state using the target network.  Then it uses this to calculate the target q values for the current state using the Bellman equation.

The loss is then calculated using the mean squared error between the expected q values and the target q values.  Finally it synchronize the target network with the q network if enough updates have passed.


### Training loop
The outer loop is over episodes, the inner loop over steps.  At each step the agent chooses an action, takes an action, and then learns from its experience.  Epsilon is decayed after each episode.


### Evaluation loop
Once training is done the model can be evaluated with a smaller, fixed epsilon value.  This makes it mostly greedy but still allows for a minimum of stochasticity.  There is also no need for a memory buffer as the agent is not learning from its experience.
