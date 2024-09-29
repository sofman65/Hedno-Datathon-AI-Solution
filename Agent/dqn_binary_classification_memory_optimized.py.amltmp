"""
Memory favored optimization in cost of processing speed
Approximate memory reduction: ~40%
"""

import os
import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
import dill
import gc


class BinaryClassificationEnv:

    def __init__(self, data, labels):
        """
        Initialize the environment.

        Args:
            data (numpy.ndarray): The input data.
            labels (numpy.ndarray): The corresponding labels.
        """

        self.data = data
        self.labels = labels
        self.state_space = self.data.shape[1]
        self.action_space = 2
        self.current_state = 0  # Initialize to first data point

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            numpy.ndarray: The next state.
            float: The reward obtained for taking the action.
        """
        # Determine the reward based on the action
        if action == self.labels[self.current_state]:

            if self.labels[self.current_state] == 0:

                reward = 3

            elif self.labels[self.current_state] == 1:

                reward = 7

        else:

            if self.labels[self.current_state] == 0:

                reward = -10

            elif self.labels[self.current_state] == 1:
                
                reward = -14

        # Choose the next state randomly
        self.current_state = random.randint(0, len(self.data)-1)

        # Return the next state and reward
        next_state = self.data[self.current_state]
        return next_state, reward


class PrioritizedReplayBuffer:

    def __init__(self, buffer_size, priority_exponent=0.6, importance_sampling_exponent=0.4):

        self.buffer_size = buffer_size
        self.priority_exponent = priority_exponent
        self.importance_sampling_exponent = importance_sampling_exponent
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.probabilities = deque(maxlen=buffer_size)
        self.epsilon = 0.01
        self.num_added = 0

    def add(self, state, action, reward, next_state, done):

        max_priority = np.max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        self.num_added += 1

    def sample(self, batch_size):

        if self.num_added < self.buffer_size:

            raise ValueError('Not enough experiences in buffer')

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.priority_exponent / \
            np.sum(priorities ** self.priority_exponent)

        indices = np.random.choice(
            len(self.buffer), size=batch_size, p=probabilities)

        states = np.array([self.buffer[idx][0] for idx in indices])
        actions = np.array([self.buffer[idx][1] for idx in indices])
        rewards = np.array([self.buffer[idx][2] for idx in indices])
        next_states = np.array([self.buffer[idx][3] for idx in indices])
        dones = np.array([self.buffer[idx][4] for idx in indices])

        weights = (len(self.buffer) *
                   probabilities[indices]) ** (-self.importance_sampling_exponent)
        weights /= np.max(weights)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):

        for i, e in zip(indices, errors):

            self.priorities[i] = np.abs(e) + self.epsilon

    def __len__(self):

        return len(self.buffer)


class DQN:

    def __init__(self, state_space, action_space, learning_rate, discount_factor, buffer_size=1000, priority_exponent=0.6, importance_sampling_exponent=0.4):
        """
        Initialize the DQN agent.

        Args:
            state_space (int): The size of the state space.
            action_space (int): The size of the action space.
            learning_rate (float): The learning rate for the optimizer.
            discount_factor (float): The discount factor for the Bellman equation.
            buffer_size (int): The size of the replay buffer.
            priority_exponent (float): The priority exponent.
            importance_sampling_exponent (float): The importance sampling exponent.
        """

        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.loss_fn = 'mse'
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.policy_net = self._build_model()
        self.target_net = self._build_model()
        self.target_net.set_weights(self.policy_net.get_weights())
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=buffer_size, priority_exponent=priority_exponent, importance_sampling_exponent=importance_sampling_exponent)

    def _build_model(self):
        """
        Build the neural network model.

        Returns:
            tensorflow.keras.models.Sequential: The neural network model.
        """

        model = Sequential([
            Dense(128, input_dim=self.state_space, activation='sigmoid'),
            Dense(self.action_space, activation='sigmoid')
        ])
        model.compile(loss=self.loss_fn, optimizer=self.optimizer)

        return model

    def update_target_network(self):
        """
        Update the target network with the weights from the policy network.
        """

        self.target_net.set_weights(self.policy_net.get_weights())

    def choose_action(self, state, epsilon):
        """
        Choose an action to take.

        Args:
            state (numpy.ndarray): The current state.
            epsilon (float): The probability of choosing a random action.

        Returns:
            int: The action to take.
        """

        if random.random() > epsilon:

            state_tensor = np.reshape(state, (1, self.state_space))
            q_values = self.policy_net.predict(state_tensor)
            action = np.argmax(q_values[0])

        else:

            action = random.randint(0, self.action_space-1)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer with its priority.

        Args:
            state (numpy.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self, batch_size, num_episodes):
        """
        Train the agent on a batch of experiences.

        Args:
            batch_size (int): The batch size to use for training.
            num_episodes (int): The number of episodes to train for.
        """

        for episode in range(num_episodes):

            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
                batch_size)

            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.bool)
            weights = np.array(weights, dtype=np.float32)

            current_q_values = self.policy_net.predict(states, verbose=0)
            current_q_values = current_q_values[np.arange(
                len(actions)), actions]

            next_q_values = self.target_net.predict(next_states, verbose=0)
            next_q_values = np.max(next_q_values, axis=-1)
            next_q_values[dones] = 0.0

            # In-place update of target_q_values
            target_q_values = np.empty_like(rewards)
            np.add(rewards, self.discount_factor *
                   next_q_values, out=target_q_values)

            # Train the model
            class_weights = {i: w for i, w in enumerate(weights)}
            self.policy_net.fit(states, target_q_values,
                                class_weight=class_weights, epochs=1, verbose=0)

            # Update the priorities of the samples in the replay buffer
            errors = np.abs(target_q_values - current_q_values)
            self.replay_buffer.update_priorities(indices, errors)

            # Clean errors
            del errors
            gc.collect()  # Force garbage collection

            if episode % 10 == 0:

                self.update_target_network()

    def predict(self, states):
        """
        Predict the Q-values for each action in the given state.

        Args:
        - states: numpy array representing the current state of the environment

        Returns:
        - q_values: numpy array representing the Q-values for each action in the given state
        """
        q_values = self.policy_net.predict(states, verbose=0)
        return q_values
        
    def save_agent(self, file_path):
        """
        Save the DQN agent to a file.

        Args:
            file_path (str): The path to the directory where the agent should be saved.
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        policy_net_file = os.path.join(file_path, "policy_net.h5")
        target_net_file = os.path.join(file_path, "target_net.h5")
        agent_file = os.path.join(file_path, "agent.pkl")

        self.policy_net.save(policy_net_file)
        self.target_net.save(target_net_file)

        with open(agent_file, 'wb') as f:
            # Set the models to None temporarily to avoid saving them with dill
            policy_net_backup = self.policy_net
            target_net_backup = self.target_net
            self.policy_net = None
            self.target_net = None

            dill.dump(self, f)

            # Restore the models
            self.policy_net = policy_net_backup
            self.target_net = target_net_backup

    @classmethod
    def load_agent(cls, file_path):
        """
        Load the DQN agent from a file.

        Args:
            file_path (str): The path to the directory where the agent is saved.

        Returns:
            DQN: The loaded DQN agent.
        """
        with open(os.path.join(file_path, "agent.pkl"), 'rb') as f:
            agent = dill.load(f)

        agent.policy_net = load_model(os.path.join(file_path, "policy_net.h5"))
        agent.target_net = load_model(os.path.join(file_path, "target_net.h5"))

        return agent


class DoubleDQN:

    def __init__(self, state_space, action_space, learning_rate, discount_factor, buffer_size=1000, priority_exponent=0.6, importance_sampling_exponent=0.4):
        """
        Initialize the DQN agent.

        Args:
            state_space (int): The size of the state space.
            action_space (int): The size of the action space.
            learning_rate (float): The learning rate for the optimizer.
            discount_factor (float): The discount factor for the Bellman equation.
            buffer_size (int): The size of the replay buffer.
            priority_exponent (float): The priority exponent.
            importance_sampling_exponent (float): The importance sampling exponent.
        """

        # Initialize Double DQN model with state_space, action_space, learning rate(learning_rate), and discount factor(discount_factor)
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.loss_fn = 'mse'
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.policy_net = self._build_model()
        self.target_net = self._build_model()
        self.target_net.set_weights(self.policy_net.get_weights())
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=buffer_size, priority_exponent=priority_exponent, importance_sampling_exponent=importance_sampling_exponent)

    def _build_model(self):
        """
        Build the neural network model.

        Returns:
            tensorflow.keras.models.Sequential: The neural network model.
        """

        model = Sequential([
            Dense(128, input_dim=self.state_space, activation='sigmoid'),
            Dense(self.action_space, activation='sigmoid')
        ])
        model.compile(loss=self.loss_fn, optimizer=self.optimizer)
        return model

    def update_target_network(self):
        """
        Update the target network with the weights of the policy network.

        Returns: None
        """

        # Update the weights of the target network with the weights of the policy network
        self.target_net.set_weights(self.policy_net.get_weights())

    def choose_action(self, state, epsilon):
        """
        Choose an action to take in the current state based on an epsilon-greedy policy.

        Args:
        - state: numpy array representing the current state of the environment
        - epsilon: float representing the probability of choosing a random action instead of the best action

        Returns:
        - action: integer representing the chosen action to take
        """

        # Epsilon-greedy strategy for choosing action
        if random.random() > epsilon:

            # Choose action with the highest Q-value predicted by the policy network
            state_tensor = np.reshape(state, (1, self.state_space))
            q_values = self.policy_net.predict(state_tensor)
            action = np.argmax(q_values[0])

        else:

            # Choose a random action with probability epsilon
            action = random.randint(0, self.action_space-1)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer with its priority.

        Args:
            state (numpy.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self, batch_size, num_episodes):
        """
        Train the agent on a batch of experiences.

        Args:
        - batch_size: integer representing the size of each batch of experiences to use during training
        - num_episodes: integer representing the number of episodes to train the agent for

        Returns: None
        """

        # Train the model using experience replay
        for episode in range(num_episodes):

            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
                batch_size)

            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.bool)
            weights = np.array(weights, dtype=np.float32)

            # Calculate the target Q-values using the Bellman equation
            # by predicting the Q-values of the next state with the target network
            # and selecting the action with the highest Q-value using the policy network
            next_q_values = self.target_net.predict(next_states, verbose=0)
            next_q_values_policy = self.policy_net.predict(
                next_states, verbose=0)
            next_q_values = next_q_values_policy[np.arange(
                len(next_q_values)), np.argmax(next_q_values, axis=-1)]
            next_q_values[dones] = 0.0

            # In-place update of target_q_values
            target_q_values = np.empty_like(rewards)
            np.add(rewards, self.discount_factor *
                   next_q_values, out=target_q_values)

            # Get the predicted Q-values for the current states and the chosen actions
            current_q_values = self.policy_net.predict(states, verbose=0)
            current_q_values = current_q_values[np.arange(
                len(actions)), actions]

            # Update the priorities of the samples in the replay buffer
            errors = np.abs(target_q_values - current_q_values)
            self.replay_buffer.update_priorities(indices, errors)

            # Clean errors
            del errors
            gc.collect()  # Force garbage collection

            # Train the policy network by minimizing the mean squared error
            # between the predicted Q-values and the target Q-values
            self.policy_net.fit(states, target_q_values,
                                sample_weight=weights, epochs=1, verbose=0)

            # Update the target network every 10 episodes
            if episode % 10 == 0:
                self.update_target_network()

    # Predict the Q-values of a state using the policy network

    def predict(self, states):
        """
        Predict the Q-values for each action in the given state.

        Args:
        - states: numpy array representing the current state of the environment

        Returns:
        - q_values: numpy array representing the Q-values for each action in the given state
        """
        # Define a function to convert Python float objects to NumPy float32
        def to_numpy_float32(x):
            return np.float32(x) if isinstance(x, float) else x

        # Vectorize the conversion function to apply it to each element in the array
        to_numpy_float32_vectorized = np.vectorize(to_numpy_float32)

        # Apply the conversion function to the input states array
        states = to_numpy_float32_vectorized(states)

        q_values = self.policy_net.predict(states, verbose=0)
        return q_values

    def save_agent(self, file_path):
        """
        Save the DQN agent to a file.

        Args:
            file_path (str): The path to the directory where the agent should be saved.
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        policy_net_file = os.path.join(file_path, "policy_net.h5")
        target_net_file = os.path.join(file_path, "target_net.h5")
        agent_file = os.path.join(file_path, "agent.pkl")

        self.policy_net.save(policy_net_file)
        self.target_net.save(target_net_file)

        with open(agent_file, 'wb') as f:
            # Set the models to None temporarily to avoid saving them with dill
            policy_net_backup = self.policy_net
            target_net_backup = self.target_net
            self.policy_net = None
            self.target_net = None

            dill.dump(self, f)

            # Restore the models
            self.policy_net = policy_net_backup
            self.target_net = target_net_backup

    @classmethod
    def load_agent(cls, file_path):
        """
        Load the DQN agent from a file.

        Args:
            file_path (str): The path to the directory where the agent is saved.

        Returns:
            DQN: The loaded DQN agent.
        """
        with open(os.path.join(file_path, "agent.pkl"), 'rb') as f:

            agent = dill.load(f)

        agent.policy_net = load_model(os.path.join(file_path, "policy_net.h5"))
        agent.target_net = load_model(os.path.join(file_path, "target_net.h5"))

        return agent
