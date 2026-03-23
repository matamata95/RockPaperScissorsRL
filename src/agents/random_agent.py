import random


class RandomAgent:
    def __init__(self, seed=None):
        if seed is not None:
            self.seed = random.seed(seed)

    def select_action(self, history, available_actions):
        return random.choice(available_actions)
