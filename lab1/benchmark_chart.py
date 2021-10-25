import numpy as np
import matplotlib.pyplot as plt
import json


def readFromJson(path):
    with open(path) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    data = readFromJson(".benchmarks/data.json")["steepest_descent_method"]

    fig, ax = plt.subplots()
    Xs = list(set([benchmark["point"][0] for benchmark in data["benchmarks"]]))
    Ys = list(set([benchmark["point"][1] for benchmark in data["benchmarks"]]))
    iterations = [benchmark["iterations"] for benchmark in data["benchmarks"]]
    width = len(Xs)

    def chunks(l, size):
        return [l[i:i+size] for i in range(0, len(l), size)]

    iterations = np.array(chunks(iterations, width))

    im = ax.imshow(iterations)
    ax.set_xticks(np.arange(len(Xs)))
    ax.set_yticks(np.arange(len(Ys)))
    ax.set_xticklabels(Xs)
    ax.set_yticklabels(Ys)

    for i in range(len(Ys)):
        for j in range(len(Xs)):
            text = ax.text(j, i, iterations[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("Iterations")
    fig.tight_layout()
    plt.show()
