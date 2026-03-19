import pickle
import numpy as np
from RPS_env import RPS_env


def human_input():
    player_input = input("Enter your move (rock, paper, scissors) "
                         "or 'exit' to quit: ").strip().lower()
    return player_input


def state_to_key(state1, state2):
    """
    Converting the state to a unique key for Q-table indexing.
    States have values:
        0 for Rock,
        1 for Paper,
        2 for Scissors.
    """
    return state1 * 1 + state2 * 3


def human_vs_q(q_path="q_table_AIvsAI.pkl"):
    try:
        with open(q_path, "rb") as f:
            q_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Q-table file '{q_path}' not found. "
              f"Please train the agent first.")
        return

    env = RPS_env()
    player_1_counter = 0
    player_2_counter = 0
    num_of_ties = 0

    while True:
        player_input = human_input()
        if player_input == "exit":
            print("Exiting the game. Thanks for playing!")
            break

        move_mapping = {"rock": 0, "paper": 1, "scissors": 2}
        if player_input not in move_mapping:
            print("Invalid input. Enter 'rock', 'paper', or 'scissors'.")
            continue

        human_move = move_mapping[player_input]
        previous_state = env.get_previous_observation()
        try:
            state_key = state_to_key(previous_state[0], previous_state[1])
            q_values = q_dict.get(state_key, [0, 0, 0])
            ai_move = int(np.argmax(q_values))  #
        except Exception:
            print("First move is random, due to not having previous state.")
            ai_move = int(np.random.choice([0, 1, 2]))

        obs, reward, done, info = env.step(ai_move, human_move)
        env.render()
        if done:
            winner_info = info.get("winning player ", "No winner info")
            if winner_info == 1:
                player_1_counter += 1
                print("Game over! AI agent wins!")
            elif winner_info == 2:
                player_2_counter += 1
                print("Game over! Human player wins!")

            print(f"Score: AI agent wins - {player_1_counter}, "
                  f"Human player wins - {player_2_counter}, "
                  f"Ties - {num_of_ties}")
            env.reset()
        else:
            num_of_ties += 1
            print("Game over! It's a tie!")


if __name__ == "__main__":
    human_vs_q()
