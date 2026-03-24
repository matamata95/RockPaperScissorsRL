import pickle
import numpy as np
from RPS_env import RPS_env
from agents.q_learning_agent import QLearningAgent


BEST_OF_GAMES = 3
BUFFER_SIZE = 5
PLAYER = -1  # Train as Player 2 to reverse reward structure


def train(episodes=2000,
          alpha=0.1,
          gamma=0.99,
          epsilon=0.1,
          q_path="tables/q_table.pkl"):
    env = RPS_env(best_of_games=BEST_OF_GAMES,
                  buffer_size=BUFFER_SIZE,
                  player=PLAYER)
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

    try:
        with open(q_path, "rb") as f:
            q_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Q-table file '{q_path}' not found.")
        return

    for ep in range(1, episodes + 1):
        env.reset()
        history = env.get_history()
        done = False
        while not done:
            try:
                history_key = QLearningAgent.history_to_key(history)
                q_values = q_dict.get(history_key, [0, 0, 0])
                if np.allclose(q_values, q_values[0]):
                    # If all Q-values are similar, choose randomly
                    ai_move = int(np.random.choice(env.available_actions()))
                else:
                    ai_move = int(np.argmax(q_values))  #
            except Exception:
                # First move is random, due to not having previous state.
                ai_move = int(np.random.choice([0, 1, 2]))

            # Player 1 selects an action using the Q-learning agent
            a0 = ai_move
            a1 = agent.select_action(history, env.available_actions())
            next_state = [a0, a1]

            history, r, terminated, info = env.step(next_state)

            if terminated:
                agent.update(history, a0, r, terminated)
                break
            # When the game is tied the game continues
            agent.update(history, a0, r,
                         terminated, next_state, env.available_actions())

        if not ep % 200:
            print(f"Episode {ep}/{episodes} completed.")

    # Save Q-table
    with open("tables/q_table_AIvsAI.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)


if __name__ == "__main__":
    train(episodes=10**5)
