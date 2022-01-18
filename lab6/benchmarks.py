from qlearning import run_qlearning, run_random, evaluate_qtable, HyperParameters
from environment import Environment, reward_default, reward_alternative_1, reward_alternative_2

from multiprocessing.pool import Pool
from itertools import repeat
from statistics import mean
import json

SIZE = 12
LEARNING_RATE = 0.8
DISCOUNT_RATE = 0.85
EPSILON = 0.5
EPISODES = 100_000
STEPS = 20

EPISODES = [1000, 5000, 10_000, 100_000]
FUNCTIONS = [reward_default, reward_alternative_1, reward_alternative_2]
LEARNING_RATES = [0.3, 0.5, 1]
DISCOUNT_RATES = [0.3, 0.5, 1]

BENCHMARK_ITERATIONS = 20


def benchmark(env: Environment, params: HyperParameters):
    env.reset()
    q_table = run_qlearning(env, params)
    success_ql, path_ql = evaluate_qtable(env, q_table, 50)
    success_rand, path_rand = run_random(env, 50)

    count_ql, count_rand = 0, 0
    count_ql += success_ql > success_rand
    count_rand += success_ql < success_rand
    if success_ql and success_rand:
        count_ql += path_ql < path_rand
        count_rand += path_ql > path_rand
    return count_ql, count_rand


def benchmark_episodes_reward_functions(env: Environment):
    counts = {function.__name__: {episodes: {"QL": 0, "RAND": 0} for episodes in EPISODES} for function in FUNCTIONS}
    for function in FUNCTIONS:
        print(f"{function.__name__=}")
        env.reward = function
        for episodes in EPISODES:
            print(f"{episodes=}")
            params = HyperParameters(episodes=episodes)
            count = [benchmark(env, params) for _ in range(BENCHMARK_ITERATIONS)]
            for elem in count:
                counts[function.__name__][episodes]["QL"] += elem[0]
                counts[function.__name__][episodes]["RAND"] += elem[1]
    return counts


def benchmark_episodes_learning_rates(env: Environment):
    counts = {learning_rate: {episodes: {"QL": 0, "RAND": 0} for episodes in EPISODES} for learning_rate in LEARNING_RATES}
    for learning_rate in LEARNING_RATES:
        print(f"{learning_rate=}")
        for episodes in EPISODES:
            print(f"{episodes=}")
            params = HyperParameters(learning_rate=learning_rate, episodes=episodes)
            count = [benchmark(env, params) for _ in range(BENCHMARK_ITERATIONS)]
            for elem in count:
                counts[learning_rate][episodes]["QL"] += elem[0]
                counts[learning_rate][episodes]["RAND"] += elem[1]
    return counts


def benchmark_episodes_discount_rates(env: Environment):
    counts = {discount_rate: {episodes: {"QL": 0, "RAND": 0} for episodes in EPISODES} for discount_rate in DISCOUNT_RATES}
    for discount_rate in DISCOUNT_RATES:
        print(f"{discount_rate=}")
        for episodes in EPISODES:
            print(f"{episodes=}")
            params = HyperParameters(discount_rate=discount_rate, episodes=episodes)
            count = [benchmark(env, params) for _ in range(BENCHMARK_ITERATIONS)]
            for elem in count:
                counts[discount_rate][episodes]["QL"] += elem[0]
                counts[discount_rate][episodes]["RAND"] += elem[1]
    return counts


def write_to_json(data, path):
    f = open(path, "w")
    json.dump(data, f, indent=2)
    f.close()


env = Environment(side_length=SIZE)
write_to_json(benchmark_episodes_reward_functions(env), ".benchmarks/reward_functions.json")
write_to_json(benchmark_episodes_learning_rates(env), ".benchmarks/learning_rates.json")
write_to_json(benchmark_episodes_discount_rates(env), ".benchmarks/discount_rates.json")