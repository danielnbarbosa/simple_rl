### Model
This is a straightforward multi-layer perceptron with RELU activations on the hidden layers.  It uses two layers, which is just enough to be considered deep, and adequate for low dimensional state spaces.  It outputs the predicted Q values for all actions.


### Experience Replay Buffer
The experience replay buffer is a circular buffer of fixed length.  This is one of two important tweaks that help the algorithm converge more reliably.  Instead of training using the immediate interactions with the environment, the agent randomly samples from the buffer.  This helps break any temporal correlations that could cause the model to overfit as well as allowing it to learn from past experiences several times.

The replay buffer is a deque of named tuples.  Each tuple consists of (state, action, reward, next_state, done).  The experience sampling is done with a fixed batch size and then converted to tensors for feeding to the network for training.  The batch size is just like the mini batch size in supervised learning and training is done in these sized batches as well.


### Agent
The agent specifics several hyperparamaters that are reasonably sane defaults but feel free to tweak them.  Performance on a particular problem can often be improved by tweaking the hyperparameters.

During initialization the agents creates two Q networks with identical weights.  This brings us to the second important tweak that helps DQN converge, fixed Q-targets.  Due to the fact the Q learning uses estimated Q values to determine how far off it's prediction is, it can often cause the model to diverge as it gets stuck chasing its own tail.  To prevent this a separate (fixed) Q network is used with values that change more slowly than the live one.  This gives a more stable base to compare the live predictions to.

The agent then initializes an optimizer, here we use Adam as it works well in lots of situations.  And then the replay buffer.

The first method we define is act(), which defines the action an agent takes for a given state (and epsilon value).   The agent does a forward pass on the state which produces predicted action values.  Then it uses epsilon greedy to decide if it should behave greedily or randomly.

The second method is step(), which what an agent does at each time step.  Here it takes a (state, action, reward, next_state, done) tuple and stores it in the experience replay buffer.  Then it samples from the buffer and learns from it's stored experience.

Which brings us to the learn() method.  This is where the meat of the algorithm is.  First we calculate the expected q values for the current state using the live model.  Then we calculate the max q values for the next state using the fixed model.  Then we use this to calculate the target q values for the current state using the Bellman equation.  The loss is then calculated using the mean squared error between the expected q values and the target q values.  Finally we apply a soft update to the fixed model.

The soft_update() is a way to gradually change fixed model to be like to live model.  Every step it will get closer by tau.


### Training loop
The main training loop simply loops over episodes and then loops over steps.  At each step the agents chooses an action, takes an action, and then learns from its experience.  We decay epsilon after each episode.


### Evaluation loop
Once training is done it is helpful to evaluate the model without the randomizing effects of epsilon greedy.  This is similar to the training loop but sets epsilon to 0 and doesn't bother with experience replay.   You can set render to True to see the agent in action.
