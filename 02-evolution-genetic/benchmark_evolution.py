from typing import NamedTuple
import evolution
import time
import json
from statistics import mean, stdev


NUMBER_OF_VERTICES = 25
NUMBER_OF_COVERED_VERTICES = 18
PERCENT_OF_GRAPH_FULLNESS = 0.5
MUTATION_PROBABILITY = 0.05
MUTATION_PROBABILITIES = [0.01, 0.05, 0.15, 0.5]
MAX_ITERATIONS = 500
MAX_ITERATIONS_LIST = [10, 50, 100, 500, 1000]
POPULATION_SIZES = [25, 50, 100, 200, 300]
# POPULATION_SIZES = [25, 50]
POPULATION_SIZE = 50
TOURNAMENT_SIZE = 2
BENCHMARK_ITERATIONS = 15


class Result(NamedTuple):
    param: float
    min_time: float
    avg_time: float
    max_time: float
    stdev_time: float
    min_vertices: int
    avg_vertices: float
    max_vertices: int
    stdev_vertices: float

    def __str__(self) -> str:
        return f"""
        {self.param=}
        {self.min_time=}, {self.avg_time=}, {self.max_time=}, {self.stdev_time=},
        {self.min_vertices=}, {self.avg_vertices=}, {self.max_vertices=}, {self.stdev_vertices=}"""


def benchmark_population_size(edges, size):
    start = time.time()
    vertices = evolution.mfind(edges, NUMBER_OF_VERTICES, size,
                               MUTATION_PROBABILITY, TOURNAMENT_SIZE, MAX_ITERATIONS,
                               NUMBER_OF_COVERED_VERTICES)
    end = time.time()
    return vertices.vertices.count(1), end-start


def benchmark_all_population_sizes(edges, sizes):
    result = {}
    for size in sizes:
        vertices, _time = benchmark_population_size(edges, size)
        result[size] = (vertices, _time)
    return result


def benchmark_mutation_probability(edges, mutation_probability):
    start = time.time()
    vertices = evolution.mfind(edges, NUMBER_OF_VERTICES, POPULATION_SIZE,
                               mutation_probability, TOURNAMENT_SIZE, MAX_ITERATIONS,
                               NUMBER_OF_COVERED_VERTICES)
    end = time.time()
    return vertices.vertices.count(1), end-start


def benchmark_all_mutation_probabilities(edges, probabilities):
    result = {}
    for probability in probabilities:
        vertices, _time = benchmark_mutation_probability(edges, probability)
        result[probability] = (vertices, _time)
    return result


def benchmark_iterations(edges, iterations):
    start = time.time()
    vertices = evolution.mfind(edges, NUMBER_OF_VERTICES, POPULATION_SIZE,
                               MUTATION_PROBABILITY, TOURNAMENT_SIZE, iterations,
                               NUMBER_OF_COVERED_VERTICES)
    end = time.time()
    return vertices.vertices.count(1), end-start


def benchmark_all_iterations(edges, iterations_list):
    result = {}
    for iterations in iterations_list:
        vertices, _time = benchmark_mutation_probability(edges, iterations)
        result[iterations] = (vertices, _time)
    return result


def benchmark(param_list, iterations, func):
    result = {param: [] for param in param_list}
    edges = evolution.generate_percent_of_edges(
            NUMBER_OF_VERTICES, PERCENT_OF_GRAPH_FULLNESS)
    evolution.show_result([0 for _ in range(NUMBER_OF_VERTICES)],
                          edges, name='graph1')
    for _ in range(iterations):
        bench = func(edges, param_list)
        for param in param_list:
            result[param].append(bench[param])
    return result


def find_min_avg_max_for_size(result, param):
    min_time = min(result[param], key=lambda x: x[1])[1]
    max_time = max(result[param], key=lambda x: x[1])[1]
    lst_of_time = [tpl[1] for tpl in result[param]]
    avg_time = mean(lst_of_time)
    stdev_time = stdev(lst_of_time)

    min_vertices = min(result[param], key=lambda x: x[0])[0]
    max_vertices = max(result[param], key=lambda x: x[0])[0]
    lst_of_vertices = [tpl[0] for tpl in result[param]]
    avg_vertices = mean(lst_of_vertices)
    stdev_vertices = stdev(lst_of_vertices)
    return Result(param, min_time, avg_time, max_time, stdev_time,
                  min_vertices, avg_vertices, max_vertices, stdev_vertices)


def find_min_avg_max(result):
    return [find_min_avg_max_for_size(result, size) for size in result]


def save_as_json(result, path):
    result_json = {}
    for object in result:
        result_json[object.param] = {
            'min_time': object.min_time,
            'avg_time': object.avg_time,
            'max_time': object.max_time,
            'stdev_time': object.stdev_time,
            'min_vertices': object.min_vertices,
            'avg_vertices': object.avg_vertices,
            'max_vertices': object.max_vertices,
            'stdev_vertices': object.stdev_vertices,
            }
    with open(path, 'w') as fp:
        json.dump(result_json, fp, indent=2)


result = find_min_avg_max(benchmark(MAX_ITERATIONS_LIST, BENCHMARK_ITERATIONS, benchmark_all_iterations))
save_as_json(result, 'lab2/.benchmarks/smth.json')
