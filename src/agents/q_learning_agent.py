import numpy as np
from collections import defaultdict


def state_to_key(state):
    """
    Convert the state to a unique key for Q-table indexing.
    0 for Rock, 1 for Paper, 2 for Scissors.
    """
    return state[0] * 1 + state[1] * 3


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        # Initializes a Q-table as a dict where every new state added will have
        # an array of three actions (Rock, Paper, Scissors) initialized to 0.0
        self.Q = defaultdict(lambda: np.zeros(3))

    def select_action(self, previous_state, available_actions):
        if None in previous_state:
            # If previous_state is None, we are at the beginning of the game
            # so we choose a random action to start the game
            return int(np.random.choice(available_actions))

        # Convert the previous state to a key for Q-table indexing
        # since we should be using the previous state to select the action
        key = state_to_key(previous_state)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available_actions))
        qs = self.Q[key]
        return int(np.argmax(qs))

    def update(self, previous_state, action, reward,
               done, next_state=None, next_available_actions=None):
        if previous_state is None or None in previous_state:
            # If previous state is None, we are at the beginning of the game
            # and there is no update to perform
            return
        ps = state_to_key(previous_state)
        q = self.Q[ps][action]
        if done or next_state is None or None in next_state:
            target = reward
        else:
            ns = state_to_key(next_state)
            if next_available_actions is None:
                best_next = np.max(self.Q[ns])
            else:
                best_next = max(self.Q[ns][a] for a in next_available_actions)
            target = reward + self.gamma * best_next
        self.Q[ps][action] += self.alpha * (target - q)
