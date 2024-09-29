import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Agent.Utils.Debug import print_caller_info

class ModelRunner:
    def __init__(self, func, X, y, test_split_size=0.2, agent_split_size=0.5, random_states=[42]):
        self.func = func
        self.X = X
        self.y = y
        self.test_split_size = test_split_size
        self.agent_split_size = agent_split_size
        self.random_states = random_states

    def run(self, X_test):
        results = []

        for random_state in self.random_states:

            # Execute the provided function on the train and test sets
            self.func(self.X, self.y, X_test, self.agent_split_size, random_state)
            