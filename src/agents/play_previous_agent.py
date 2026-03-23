import numpy as np


class PlayPreviousAgent:
    def __init__(self, seed=None):
        if seed is not None:
            self.seed = np.random.seed(seed)

    def select_action(self, history, available_actions):
        previous_state = history[-1]  # Get the previous state
        if None in previous_state:
            # if previous_state is None, we are at the beginning of the game
            return int(np.random.choice(available_actions))

        # Play opponent's previous mode
        return previous_state[0]  # assumes that PlayPreviousAgent is Player 2
