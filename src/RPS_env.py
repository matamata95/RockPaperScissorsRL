import copy


class RPS_env:
    def __init__(self):
        self.previous_state = [None, None]
        self.reset()

    def reset(self):
        self.state = [None, None]
        self.done = False
        return self.get_observation()

    def get_observation(self):
        return copy.copy(self.state)

    def get_previous_observation(self):
        return copy.copy(self.previous_state)

    def available_actions(self):
        return [0, 1, 2]

    def action_mapping(self, action):
        if action == "rock":
            return 0
        if action == "paper":
            return 1
        if action == "scissors":
            return 2

    def step(self, action0, action1):
        if self.done:
            raise ValueError("Game is over.")

        if action0 in ["rock", "paper", "scissors"]:
            action0 = self.action_mapping(action0)
        if action1 in ["rock", "paper", "scissors"]:
            action1 = self.action_mapping(action1)

        if action0 not in list(range(3)) or action1 not in list(range(3)):
            raise ValueError("Invalid action. Choose values from "
                             "[0, 1, 2] or ['rock', 'paper', 'scissors'].")

        self.previous_state = copy.copy(self.state)
        self.state[0] = action0  # Player 1 action
        self.state[1] = action1  # Player 2 action

        winner = self._check_winner()
        if winner == 1:
            self.done = True
            reward = 1
            info = {"winning player ": winner}
            return self.get_observation(), reward, self.done, info
        elif winner == 2:
            self.done = True
            reward = -1
            info = {"winning player ": winner}
            return self.get_observation(), reward, self.done, info
        elif winner == 0:
            reward = -0.05  # Small penalty for each move
            info = {"winning player ": winner}
            return self.get_observation(), reward, self.done, info
        raise ValueError("Unexpected error in determining the winner.")

    def _check_winner(self):
        """
        Determines the winner of the game.
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

    def state_to_key(self):
        """
        Converting the state to a unique key for Q-table indexing.
        0 for Rock, 1 for Paper, 2 for Scissors.
        """
        return self.state[0] * 1 + self.state[1] * 3

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
