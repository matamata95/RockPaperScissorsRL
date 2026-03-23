import numpy as np


class WinStayAgent:
    def __init__(self, seed=None):
        if seed is not None:
            self.seed = np.random.seed(seed)

    def _check_winner(self, history):
        """
        Determines the winner. Best of buffer_size games are played.
            Returns:
                0 if it's a tie,
                1 if Player 1 wins,
                2 if Player 2 wins.
            Inputs:
                history[-1][0]: Player 1 action
                history[-1][1]: Player 2 action
            Actions:
                0: Rock -> Beats Scissors, Loses to Paper
                1: Paper -> Beats Rock, Loses to Scissors
                2: Scissors -> Beats Paper, Loses to Rock
        """
        p1, p2 = history[-1]
        if p1 == p2:
            return 0  # Tie
        elif (p1 == 0 and p2 == 2) or \
             (p1 == 1 and p2 == 0) or \
             (p1 == 2 and p2 == 1):
            return 1  # Player 1 wins
        else:
            return 2  # Player 2 wins

    def select_action(self, history, available_actions):
        # Assumes that WinStayAgent is Player 2
        if self._check_winner(history) == 1:
            # If the agent lost the previous round, # it will switch to the
            # action that beats the opponent's previous action
            return history[-1][0]
        elif self._check_winner(history) == 2:
            # If the agent won the previous round,
            # it will repeat the same action
            return history[-1][1]
        else:
            # If the previous round was a tie, the agent will choose a random
            # action from the available actions
            return int(np.random.choice(available_actions))
