from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeVar, Generic, Callable


Shape = TypeVar("Shape")
DType = TypeVar("DType")
np.random.seed(0)


class Matrix(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        inputs: Matrix['60000, 784', np.float]
    """
    pass


class HiddenLayersNumberError(Exception):
    pass


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs, activation_function=None):
        output = np.dot(inputs, self.weights) + self.biases
        return output[0] if not activation_function else activation_function(output[0])

    def backward(self, inputs: Matrix['N,1', float], delta: Matrix['N,1', float], learn_rate):
        self.weights += -learn_rate * inputs @ delta.T
        self.biases += -learn_rate * delta.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def sigmoid_derivative(s):
    return s*(1-s)


def quadratic_cost(output_layer, label):
    return 1 / len(output_layer) * np.sum((output_layer - label) ** 2, axis=0)


def quadratic_cost_derivative(output_layer, label):
    return output_layer - label


class Network:
    def __init__(self, inputs_batch: Matrix['H, W', float],
                 labels: Matrix['H, 10', int],
                 input_layer: Layer, hidden_layer: Layer,
                 hidden_layer_activation_function,
                 output_layer_activation_function,
                 hidden_layer_activation_function_derivative,
                 cost_function_derivative) -> None:
        self.inputs_batch = inputs_batch
        self.labels = labels
        self.iLayer_obj = input_layer
        self.hLayer_obj = hidden_layer
        self.hLayer_activation = hidden_layer_activation_function
        self.oLayer_activation = output_layer_activation_function
        self.hLayer_dActivation = hidden_layer_activation_function_derivative
        self.dCost = cost_function_derivative

    def forward(self, inputs):
        input_layer = self.iLayer_obj.forward(inputs, self.hLayer_activation)
        hidden_layer = self.hLayer_obj.forward(input_layer, self.oLayer_activation)
        return input_layer, hidden_layer

    def backward(self, inputs, label, input_layer, hidden_layer, learn_rate):
        delta_o = self.dCost(hidden_layer, label)

        input_layer.shape += (1, )
        hidden_layer.shape += (1, )
        inputs.shape += (1,)
        delta_o.shape += (1,)

        self.hLayer_obj.backward(input_layer, delta_o, learn_rate)

        delta_h = self.hLayer_obj.weights @ delta_o * self.hLayer_dActivation(input_layer)
        self.iLayer_obj.backward(inputs, delta_h, learn_rate)

    @staticmethod
    def calculate_accuracy(predictions, labels):
        correct = list(filter(
            lambda elem: elem[0] == np.argmax(elem[1]), zip(predictions, labels)
            ))
        return round(len(correct)/len(predictions)*100, 2)

    def run(self, learn_rate: float, epochs: int) -> list[int]:
        for _ in range(epochs):
            predictions = []
            for inputs, label in zip(self.inputs_batch, self.labels):
                input_layer, hidden_layer = self.forward(inputs)
                predictions.append(int(np.argmax(hidden_layer)))
                self.backward(inputs, label, input_layer, hidden_layer, learn_rate)
            print("Accuracy: ", self.calculate_accuracy(predictions, self.labels), "%")
        return predictions


if __name__ == "__main__":
    images, labels = get_mnist()

    learn_rate = 0.01
    layer1_obj = Layer(784, 20)
    layer2_obj = Layer(20, 10)
    # layer3_obj = Layer(20, 10)
    epochs = 3
    network = Network(images, labels, layer1_obj, layer2_obj, sigmoid, softmax,
                      sigmoid_derivative, quadratic_cost_derivative)
    predictions = network.run(learn_rate, epochs)

    # while True:
    #     index = int(input("Enter a number (0 - 59999): "))
    #     img = images[index]
    #     plt.imshow(img.reshape(28, 28), cmap="Greys")
    #     layer1 = layer1_obj.forward(img, sigmoid)
    #     layer3 = layer2_obj.forward(layer1, sigmoid)
    #     plt.title(f"Prediction: {layer2.argmax()}")
    #     plt.show()