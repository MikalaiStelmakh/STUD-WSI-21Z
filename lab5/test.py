from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def forward(weights, bayeses, inputs):
    layers = [sigmoid(bayeses[0] + weights[0] @ inputs)]
    layers += [sigmoid(bayes + weights @ layers[i])
                      for i, (bayes, weights) in enumerate(zip(bayeses[1:-1], weights[1:-1]))]
    layers.append(softmax(bayeses[-1] + weights[-1] @ layers[-1]))
    return layers


def backward(layers, weights, bayeses, img, l, learn_rate):
    delta = layers[-1] - l
    weights[-1] += -learn_rate * delta @ np.transpose(layers[-2])
    bayeses[-1] += -learn_rate * delta

    for (i_w, _), (i_l, _) in zip(enumerate(weights[:0:-1]), enumerate(layers[-2:0:-1])):
        i_w = len(weights) - 1 - i_w
        i_l = len(layers) - 2 - i_l
        delta = np.transpose(weights[i_w]) @ delta * sigmoid_derivative(layers[i_l])
        weights[i_w-1] += -learn_rate * delta @ np.transpose(layers[i_l-1])
        bayeses[i_w-1] += -learn_rate * delta

    delta = np.transpose(weights[1]) @ delta * sigmoid_derivative(layers[0])
    weights[0] += -learn_rate * delta @ np.transpose(img)
    bayeses[0] += -learn_rate * delta

    return weights, bayeses


def calculate_accuracy(predictions, labels, decimals):
    correct = list(filter(
            lambda elem: elem[0] == np.argmax(elem[1]), zip(predictions, labels)
            ))
    return round(len(correct)/len(predictions)*100, decimals)


def run(images, labels, weights, bayeses, epochs, learn_rate):
    for _ in range(epochs):
        predictions = []
        for img, l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)
            layers = forward(weights, bayeses, img)
            predictions.append(np.argmax(layers[-1]))
            weights, bayeses = backward(
                layers, weights, bayeses, img, l, learn_rate
                )
        print(f"Acc: {calculate_accuracy(predictions, labels, 2)}%")
    return predictions


if __name__ == "__main__":
    images, labels = get_mnist()
    w_i_h = 0.01 * np.random.randn(784, 20).T
    w_h_h = 0.01 * np.random.randn(20, 20).T
    w_h_o = 0.01 * np.random.randn(20, 10).T
    b_i_h = np.zeros((20, 1))
    b_h_h = np.zeros((20, 1))
    b_h_o = np.zeros((10, 1))

    learn_rate = 0.01
    nr_correct = 0
    epochs = 5
    weights = [w_i_h, w_h_o]
    bayeses = [b_i_h, b_h_o]
    run(images, labels, weights, bayeses, epochs, learn_rate)