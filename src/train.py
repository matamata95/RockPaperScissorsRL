import pickle
import random
from RPS_env import RPS_env
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.play_previous_agent import PlayPreviousAgent
from agents.win_stay_agent import WinStayAgent


def action_mapping(action):
    if action == 0:
        return "ROCK"
    if action == 1:
        return "PAPER"
    if action == 2:
        return "SCISSORS"


def train(episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1,
          best_of_games=3, buffer_size=5, verbose=False):

    env = RPS_env(best_of_games=best_of_games, buffer_size=buffer_size)
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    opponent_classes = [RandomAgent, PlayPreviousAgent, WinStayAgent]


# Need to implement a reset after X amount of games
    for ep in range(1, episodes + 1):
        env.reset()
        opp = random.choice(opponent_classes)()  # Randomly select an opp agent
        done = False
        for i in range(buffer_size):
            round = 1
            history = env.get_history()
            while not done:
                # Player 1 selects an action using the Q-learning agent
                a0 = agent.select_action(history, env.available_actions())
                # Player 2 selects an action using the opponent agent
                a1 = opp.select_action(history, env.available_actions())

                next_state = [a0, a1]

                history, r, done, info = env.step(next_state)

                if verbose:
                    print(f"Episode {ep}, Round {round}")
                    print(f"Agent action: {action_mapping(a0)} || "
                          f"Opponent action: {action_mapping(a1)}")
                    print(f"Previous state: {list(history)}, \n"
                          f"Reward: {r}, \n"
                          f"Game state: {info}")
                    round += 1

                if done:
                    agent.update(history, a0, r, done)
                    break
                # When the game is tied the game continues
                agent.update(history, a0, r,
                             done, next_state, env.available_actions())

            if not ep % 200:
                print(f"Episode {ep}/{episodes} completed.")

    # Save Q-table
    with open("tables/q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)


if __name__ == "__main__":
    train(episodes=10**5, verbose=False)
