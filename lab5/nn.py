from typing import TypeVar
from data import get_mnist
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


np.random.seed(0)

NDArrayFloat = npt.NDArray[np.floating]
ScalarOrArray = TypeVar("ScalarOrArray", float, np.floating, NDArrayFloat)


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.weights: ScalarOrArray = np.random.uniform(-0.5, 0.5, (n_outputs, n_inputs))
        self.biases: ScalarOrArray = np.zeros((n_outputs, 1))


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def forward(layers: list[Layer], inputs: ScalarOrArray) -> list[ScalarOrArray]:
    """Forward propagation.
    Modifies the inputs of each layer using weights, biases and activation function.
    Returns these modified inputs as an outputs."""
    # Outputs of the input layer
    layers_outputs = [sigmoid(layers[0].biases + layers[0].weights @ inputs)]

    # Outputs of all the hidden layers
    for i, (biases, weights) in enumerate([(layer.biases, layer.weights) for layer in layers[1:-1]]):
        layers_outputs += [sigmoid(biases + weights @ layers_outputs[i])]

    # Outputs of the output layer
    layers_outputs.append(softmax(layers[-1].biases + layers[-1].weights @ layers_outputs[-1]))
    return layers_outputs


def backward(layers: list[Layer], layers_outputs: list[ScalarOrArray],
             inputs: ScalarOrArray, labels: ScalarOrArray, learn_rate: float) \
             -> list[Layer]:

    # Modifying weights and biases of the output layer
    delta = layers_outputs[-1] - labels
    layers[-1].weights += -learn_rate * delta @ np.transpose(layers_outputs[-2])
    layers[-1].biases += -learn_rate * delta

    # Modifying weights and biases of all the hidden layers
    for (i_w, _), (i_l, _) in zip(enumerate([layer.weights for layer in layers[:0:-1]]), enumerate(layers_outputs[-2:0:-1])):
        i_w = len(layers) - 1 - i_w
        i_l = len(layers_outputs) - 2 - i_l
        delta = np.transpose(layers[i_w].weights) @ delta * sigmoid_derivative(layers_outputs[i_l])
        layers[i_w-1].weights += -learn_rate * delta @ np.transpose(layers_outputs[i_l-1])
        layers[i_w-1].biases += -learn_rate * delta

    # Modifying weights and biases of the input layer
    delta = np.transpose(layers[1].weights) @ delta * sigmoid_derivative(layers_outputs[0])
    layers[0].weights += -learn_rate * delta @ np.transpose(inputs)
    layers[0].biases += -learn_rate * delta

    return layers


def calculate_accuracy(predictions: list[int], labels: list[int], n_decimals: int = 2) -> float:
    correct = list(filter(
            lambda elem: elem[0] == np.argmax(elem[1]), zip(predictions, labels)
            ))
    return round(len(correct)/len(predictions)*100, n_decimals)


def run(inputs_batch: ScalarOrArray, labels_batch: ScalarOrArray,
        layers: list[Layer], epochs: int, learn_rate: float) -> list[int]:
    for _ in range(epochs):
        predictions = []
        for inputs, labels in zip(inputs_batch, labels_batch):
            inputs.shape += (1,)
            labels.shape += (1,)
            layers_outputs = forward(layers, inputs)
            predictions.append(np.argmax(layers_outputs[-1]))
            layers = backward(
                layers, layers_outputs, inputs, labels, learn_rate
                )
        print(f"Acc: {calculate_accuracy(predictions, labels_batch, 2)}%")
    return predictions


if __name__ == "__main__":
    images, labels = get_mnist()

    input_layer = Layer(784, 20)
    hidden_layer1 = Layer(20, 20)
    hidden_layer2 = Layer(20, 10)

    learn_rate = 0.01
    epochs = 5
    layers = [input_layer, hidden_layer1, hidden_layer2]
    run(images, labels, layers, epochs, learn_rate)