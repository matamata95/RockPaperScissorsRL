import copy
from collections import deque


class RPS_env:
    def __init__(self, best_of_games=3, buffer_size=5, player=1):
        """
        self.player = 1 if the agent is training as Player 1.
        self.player = -1 if the agent is training as Player 2.

        As to reverse the reward structure, we can simply
        multiply the reward by -1 when the agent is Player 2.
        """
        self.best_of_games = best_of_games
        self.buffer_size = buffer_size
        self.player_1_wins = 0
        self.player_2_wins = 0
        self.player = player
        self.reset()

    def reset(self):
        # Inteded to reset after X amount of games are played
        self.state = [None, None]
        self.history = deque([[None, None]] * self.buffer_size,
                             maxlen=self.buffer_size)
        self.player_1_wins = 0
        self.player_2_wins = 0

        self.done = False
        return self.get_history()

    def get_history(self):
        return copy.copy(self.history)

    def available_actions(self):
        return [0, 1, 2]

    def action_mapping(self, action):
        if action == "rock":
            return 0
        if action == "paper":
            return 1
        if action == "scissors":
            return 2

    def step(self, state):
        if self.done:
            raise ValueError("Game is over.")

        action0 = state[0]
        action1 = state[1]

        if action0 in ["rock", "paper", "scissors"]:
            action0 = self.action_mapping(action0)
        if action1 in ["rock", "paper", "scissors"]:
            action1 = self.action_mapping(action1)

        if action0 not in list(range(3)) or action1 not in list(range(3)):
            raise ValueError(f"Invalid action. Action0: {action0}, "
                             f"Action1: {action1}. Choose values from "
                             "[0, 1, 2] or ['rock', 'paper', 'scissors'].")

        self.state[0] = action0  # Player 1 action
        self.state[1] = action1  # Player 2 action
        self.history.append(copy.copy(self.state))
        winner = self._check_winner()

        to_win = self.best_of_games // 2 + 1
        if winner == 0:
            reward = 0
            info = {"Game is tied."}
            return self.get_history(), reward, False, info

        elif winner == 1:
            self.player_1_wins += 1
            reward = 0.5 * self.player
            if self.player_1_wins >= to_win:
                self.done = True
                reward = 5 * self.player
                info = {"winning player ": winner}
                return self.get_history(), reward, True, info

        elif winner == 2:
            self.player_2_wins += 1
            reward = -0.5 * self.player
            if self.player_2_wins >= to_win:
                self.done = True
                reward = -5 * self.player
                info = {"winning player ": winner}
                return self.get_history(), reward, True, info

        # In case the game is not done, we return the current state and reward
        return self.get_history(), reward, False, {}

    def _check_winner(self):
        """
        Determines the winner. Best of buffer_size games are played.
            Returns:
                0 if it's a tie,
                1 if Player 1 wins,
                2 if Player 2 wins.
            Inputs:
                self.state[0]: Player 1 action
                self.state[1]: Player 2 action
            Actions:
                0: Rock -> Beats Scissors, Loses to Paper
                1: Paper -> Beats Rock, Loses to Scissors
                2: Scissors -> Beats Paper, Loses to Rock
        """
        p1, p2 = self.state
        if p1 == p2:
            return 0  # Tie
        elif (p1 == 0 and p2 == 2) or \
             (p1 == 1 and p2 == 0) or \
             (p1 == 2 and p2 == 1):
            return 1  # Player 1 wins
        else:
            return 2  # Player 2 wins

    def render(self):
        out = ""
        for i in self.state:
            if i == 0:
                out += """
                        _______
                    ---'   ____)
                        (_____)
                        (_____)
                        (____)
                    ---.(___)
                            Rock
                    """
            elif i == 1:
                out += """
                        _______
                    ---'   ____)____
                              ______)
                            _______)
                            _______)
                    ---.__________)
                            Paper
                    """
            elif i == 2:
                out += """
                        _______
                    ---'   ____)____
                              ______)
                         __________)
                        (____)
                    ---.(___)
                            Scissors
                    """

        print(out)
