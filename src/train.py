import pickle
from RPS_env import RPS_env
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


def train(episodes=2000,
          alpha=0.1,
          gamma=0.99,
          epsilon=0.1):
    env = RPS_env()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    opp = RandomAgent()

    for ep in range(1, episodes + 1):
        env.reset()
        previous_state = env.get_previous_observation()
        done = False
        while not done:
            # Player 1 selects an action using the Q-learning agent
            a0 = agent.select_action(previous_state, env.available_actions())
            # Player 2 selects an action using the random agent
            a1 = opp.select_action(previous_state, env.available_actions())

            # testing agent inputs
            # print(f"Previous state: {previous_state}, "
            #       f"Player 1 action: {a0}, Player 2 action: {a1}")

            next_state, r, terminated, info = env.step(a0, a1)
            if terminated:
                agent.update(next_state, a0, r, terminated)
                break
            # When the game is tied the game continues
            agent.update(previous_state, a0, r,
                         terminated, next_state, env.available_actions())
            previous_state = next_state

        if not ep % 200:
            print(f"Episode {ep}/{episodes} completed.")

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)


if __name__ == "__main__":
    train(episodes=10**5)
