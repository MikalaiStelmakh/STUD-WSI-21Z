import numpy as np
import matplotlib.pyplot as plt
import json


def readFromJson(path):
    with open(path) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    METHOD = "steepest_descent_method"
    # METHOD = "newton_method"
    data = readFromJson(".benchmarks/data.json")[METHOD]

    fig, ax = plt.subplots()
    Xs = [benchmark["point"][0] for benchmark in data["benchmarks"]]
    Ys = [benchmark["point"][1] for benchmark in data["benchmarks"]]

    def remove_duplicates(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    Xs = remove_duplicates(Xs)
    Ys = remove_duplicates(Ys)
    iterations = [benchmark["iterations"] for benchmark in data["benchmarks"]]
    width = len(Xs)

    def chunks(l, size):
        return [l[i:i+size] for i in range(0, len(l), size)]

    if METHOD == "steepest_descent_method":
        for index, _ in enumerate(iterations):
            iterations[index] /= 10000
            iterations[index] = round(iterations[index], 1)
    iterations = np.array(chunks(iterations, width))

    im = ax.imshow(iterations)
    cbar = ax.figure.colorbar(im, ax=ax)
    if METHOD == "steepest_descent_method":
        cbar.ax.set_ylabel("Iterations x 10000", rotation=-90, va="bottom")
    else:
        cbar.ax.set_ylabel("Iterations", rotation=-90, va="bottom")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
    plt.savefig("graphs/" + "iterations_" + METHOD)
