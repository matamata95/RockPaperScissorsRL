import pickle
import numpy as np
from RPS_env import RPS_env
from agents.q_learning_agent import QLearningAgent


BEST_OF_GAMES = 5
BUFFER_SIZE = 5
# Index of a human player, 0 for player 1, 1 for player 2.
HUMAN_PLAYER = 1


def human_input():
    player_input = input("Enter your move (rock, paper, scissors) "
                         "or 'exit' to quit: ").strip().lower()
    return player_input


def human_vs_q(q_path="tables/q_table_AIvsAI.pkl"):
    global HUMAN_PLAYER
    try:
        with open(q_path, "rb") as f:
            q_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Q-table file '{q_path}' not found. "
              f"Please train the agent first.")
        return

    env = RPS_env(best_of_games=BEST_OF_GAMES, buffer_size=BUFFER_SIZE)
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
        history = env.get_history()
        try:
            history_key = QLearningAgent.history_to_key(history)
            q_values = q_dict.get(history_key, [0, 0, 0])
            ai_move = int(np.argmax(q_values))  #
        except Exception:
            ai_move = int(np.random.choice([0, 1, 2]))

        if HUMAN_PLAYER:
            next_state = [ai_move, human_move]
        else:
            next_state = [human_move, ai_move]

        obs, reward, done, info = env.step(next_state)
        env.render()
        print(f"History: {history}")
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


if __name__ == "__main__":
    human_vs_q()
