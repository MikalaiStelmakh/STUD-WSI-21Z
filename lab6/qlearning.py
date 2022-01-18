import argparse
import random
from typing import NamedTuple
import numpy as np
from environment import Action, Field, Environment, GameInterface

# random.seed(1)

class HyperParameters(NamedTuple):
    """Tuple with all the possible parameters of the QLearning algorithm."""
    learning_rate: float = 0.8
    discount_rate: float = 0.85

    epsilon_max: float = 0.5
    epsilon_min: float = 1e-3
    decay_rate: float = 5e-5

    episodes: int = 100_000
    steps: int = 400


def run_qlearning(env: Environment, params: HyperParameters = HyperParameters()):
    """Runs the QLearning algorithm, filling in the Qtable and returns it as a result."""
    q_table = np.zeros([env.total_fields, len(Action)])
    epsilon = params.epsilon_max
    for episode in range(params.episodes):
        env.reset()
        state = env.state
        epochs, reward = 0, 0
        done = False
        while not done:
            if random.random() < epsilon:
                action = Action.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1-params.learning_rate) * old_value + \
                        params.learning_rate * (reward + params.discount_rate*next_max)

            q_table[state, action] = new_value

            state = next_state
            epochs += 1

    return q_table


def evaluate_qtable(env: Environment, q_table, max_steps: int):
    env.reset()
    path = []
    success = False
    done = False
    for _ in range(max_steps):
        state = env.state
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        path.append(state)
        if done:
            success = Field(env.map[env.state]) == Field.GOAL
            break
    return success, path


def run_random(env: Environment, max_steps: int):
    env.reset()
    path = []
    success = False
    done = False
    for _ in range(max_steps):
        action = Action.sample()
        state, reward, done = env.step(action)
        path.append(state)
        if done:
            success = Field(env.map[env.state]) == Field.GOAL
            break
    return success, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description =
        """QUber game with QLearning algorithm."""
    )
    parser.add_argument('--size', type=int, default=8,
                        help="Side length of the board (default: 8).")
    parser.add_argument('--start', type=int, default=None,
                        help="Agent's start position (default: 0).")
    parser.add_argument('--goal', type=int, default=None,
                        help="Agent's goal position (default: size^2 - size).")
    parser.add_argument('--p_hole', type=float, default=0.25,
                        help="Probability of hole (default: 25%%).")
    parser.add_argument('--steps', type=int, default=50,
                        help="Maximum number of agent's steps (default: 50).")
    args = parser.parse_args()

    side_length = args.size
    pos_start = args.start
    pos_goal = args.goal
    probability_of_hole = args.p_hole
    steps = args.steps

    env = Environment(side_length, pos_start, pos_goal, probability_of_hole)
    qtable = run_qlearning(env)
    success, path = evaluate_qtable(env, qtable, max_steps=steps)
    ui = GameInterface(env, path)
    ui._print_map()
