from typing import NamedTuple
from nn import Layer, run, calculate_accuracy
from data import get_mnist
import json
from statistics import mean, stdev


# Default values
IMAGES, LABELS = get_mnist()
HIDDEN_LAYERS = 2
NEURONS = 40
EPOCHS = 3
LEARN_RATE = 0.01
BENCHMARK_ITERATIONS = 10

# Parameters to benchmark
HIDDEN_LAYERS_LIST = [1, 2, 3]
NEURONS_LIST = [10, 40, 160]
EPOCHS_LIST = [1, 3, 5]
LEARN_RATE_LIST = [0.01, 0.1, 0.3]


class Result(NamedTuple):
    param: float
    min_accuracy: float
    avg_accuracy: float
    max_accuracy: float
    stdev_accuracy: float

    def __str__(self) -> str:
        return f"""
        {self.param=}
        {self.min_accuracy=}, {self.avg_accuracy=}, {self.max_accuracy=},
        {self.stdev_accuracy=}
        """


def create_layers(n_layers, n_neurons):
    layers = [Layer(784, n_neurons)]
    if n_layers > 1:
        for _ in range(n_layers-1):
            layers.append(Layer(n_neurons, n_neurons))
    layers.append(Layer(n_neurons, 10))
    return layers


def benchmark_hidden_layers_number(number):
    layers = create_layers(number, NEURONS)
    predictions = run(IMAGES, LABELS, layers, EPOCHS, LEARN_RATE)
    return calculate_accuracy(predictions, LABELS)


def benchmark_all_hidden_layers_numbers(numbers):
    result = {}
    for number in numbers:
        result[number] = benchmark_hidden_layers_number(number)
    return result


def benchmark_neurons_number(number):
    layers = create_layers(HIDDEN_LAYERS, number)
    predictions = run(IMAGES, LABELS, layers, EPOCHS, LEARN_RATE)
    return calculate_accuracy(predictions, LABELS)


def benchmark_all_neurons_numbers(numbers):
    result = {}
    for number in numbers:
        result[number] = benchmark_neurons_number(number)
    return result


def benchmark_epochs(layers, epochs):
    predictions = run(IMAGES, LABELS, layers, epochs, LEARN_RATE)
    return calculate_accuracy(predictions, LABELS)


def benchmark_all_epochs(epochs_lst):
    layers = create_layers(HIDDEN_LAYERS, NEURONS)
    result = {}
    for epochs in epochs_lst:
        result[epochs] = benchmark_epochs(layers, epochs)
    return result


def benchmark_learn_rate(layers, learn_rate):
    predictions = run(IMAGES, LABELS, layers, EPOCHS, learn_rate)
    return calculate_accuracy(predictions, LABELS)


def benchmark_all_learn_rates(learn_rates):
    layers = create_layers(HIDDEN_LAYERS, NEURONS)
    result = {}
    for learn_rate in learn_rates:
        result[learn_rate] = benchmark_learn_rate(layers, learn_rate)
    return result


def benchmark(param_list, iterations, func):
    result = {param: [] for param in param_list}
    print(func.__name__)
    for i in range(iterations):
        print("Iteration ", i)
        bench = func(param_list)
        for param in param_list:
            result[param].append(bench[param])
    return result


def find_min_avg_max_for_param(result, param):
    min_accuracy = min(result[param])
    max_accuracy = max(result[param])
    lst_of_accuracy = [tpl for tpl in result[param]]
    avg_accuracy = mean(lst_of_accuracy)
    stdev_accuracy = stdev(lst_of_accuracy)
    return Result(param, min_accuracy, avg_accuracy, max_accuracy, stdev_accuracy)


def find_min_avg_max(result):
    return [find_min_avg_max_for_param(result, param) for param in result]


def save_as_json(result, path):
    result_json = {}
    for object in result:
        result_json[object.param] = {
            'min_accuracy': object.min_accuracy,
            'avg_accuracy': object.avg_accuracy,
            'max_accuracy': object.max_accuracy,
            'stdev_accuracy': object.stdev_accuracy,
            }
    with open(path, 'w') as fp:
        json.dump(result_json, fp, indent=2)


result = find_min_avg_max(benchmark(
    HIDDEN_LAYERS_LIST, BENCHMARK_ITERATIONS, benchmark_all_hidden_layers_numbers
    ))
save_as_json(result, "lab5/.benchmarks/hidden_layers.json")

result = find_min_avg_max(benchmark(
    NEURONS_LIST, BENCHMARK_ITERATIONS, benchmark_all_neurons_numbers
    ))
save_as_json(result, "lab5/.benchmarks/neurons.json")

result = find_min_avg_max(benchmark(
    EPOCHS_LIST, BENCHMARK_ITERATIONS, benchmark_all_epochs
    ))
save_as_json(result, "lab5/.benchmarks/epochs.json")

result = find_min_avg_max(benchmark(
    LEARN_RATE_LIST, BENCHMARK_ITERATIONS, benchmark_all_learn_rates
    ))
save_as_json(result, "lab5/.benchmarks/learn_rates.json")
