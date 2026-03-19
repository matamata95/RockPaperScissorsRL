import pickle
import numpy as np
from RPS_env import RPS_env
from agents.q_learning_agent import QLearningAgent


def state_to_key(state1, state2):
    """
    Converting the state to a unique key for Q-table indexing.
    States have values:
        0 for Rock,
        1 for Paper,
        2 for Scissors.
    """
    return state1 * 1 + state2 * 3


def train(episodes=2000,
          alpha=0.1,
          gamma=0.99,
          epsilon=0.1,
          q_path="tables/q_table.pkl"):
    env = RPS_env()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

    try:
        with open(q_path, "rb") as f:
            q_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Q-table file '{q_path}' not found.")
        return

    for ep in range(1, episodes + 1):
        env.reset()
        previous_state = env.get_previous_observation()
        done = False
        while not done:
            try:
                state_key = state_to_key(previous_state[0], previous_state[1])
                q_values = q_dict.get(state_key, [0, 0, 0])
                if np.allclose(q_values, q_values[0]):
                    # If all Q-values are the same, choose randomly
                    ai_move = int(np.random.choice(env.available_actions()))
                else:
                    ai_move = int(np.argmax(q_values))  #
            except Exception:
                # First move is random, due to not having previous state.
                ai_move = int(np.random.choice([0, 1, 2]))

            # Player 1 selects an action using the Q-learning agent
            a0 = agent.select_action(previous_state, env.available_actions())
            # Player 2 selects an action using the random agent
            a1 = ai_move

            next_state, r, terminated, info = env.step(a0, a1)
            if terminated:
                agent.update(previous_state, a0, r, terminated)
                break
            # When the game is tied the game continues
            agent.update(previous_state, a0, r,
                         terminated, next_state, env.available_actions())
            previous_state = next_state

        if not ep % 200:
            print(f"Episode {ep}/{episodes} completed.")

    # Save Q-table
    with open("tables/q_table_AIvsAI.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)


if __name__ == "__main__":
    train(episodes=10**5)
