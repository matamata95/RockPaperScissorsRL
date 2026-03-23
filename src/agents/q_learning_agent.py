import numpy as np
from collections import defaultdict


def state_to_key(state):
    """
    Convert the state to a unique key for Q-table indexing.
    0 for Rock, 1 for Paper, 2 for Scissors.
    """
    return state[0] * 1 + state[1] * 3


def history_to_key(history_deque):
    """
    Converts the history of states from deque into a single tuple key
    for Q-table indexing.
        Input: deque([[0, 1], [1, 2], [2, 0]])
        Output: (0, 1, 1, 2, 2, 0)
    """
    return tuple(x for round_state in history_deque for x in round_state)


# TODO: Implement history buffer for action selection
# UPDATE state_to_key to handle all states
# REMOVE previous_state as its being replace by history buffer


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        # Initializes a Q-table as a dict where every new state added will have
        # an array of three actions (Rock, Paper, Scissors) initialized to 0.0
        self.Q = defaultdict(lambda: np.zeros(3))

    def select_action(self, history, available_actions):
        key = history_to_key(history)

        if not any(x is not None for x in key):
            # If history contains only None values, we are at the beginning of
            # the game so we choose a random action to start the game
            return int(np.random.choice(available_actions))

        # Exploration vs Exploitation
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available_actions))
        qs = self.Q[key]
        return int(np.argmax(qs))

    def update(self, history, action, reward,
               done, next_state=None, next_available_actions=None):
        history_key = history_to_key(history)
        if not any(x is not None for x in history_key):
            # If history contains only None values, we are at the beginning of
            # the game, there is nothing to update
            return

        q = self.Q[history_key][action]
        if done:
            target = reward
        else:
            history.append(next_state)
            ns = history_to_key(history)
            if next_available_actions is None:
                best_next = np.max(self.Q[ns])
            else:
                best_next = max(self.Q[ns][a] for a in next_available_actions)
            target = reward + self.gamma * best_next
        self.Q[history_key][action] += self.alpha * (target - q)
