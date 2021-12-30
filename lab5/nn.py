from typing import TypeVar
from data import get_mnist
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import argparse


NDArrayFloat = npt.NDArray[np.floating]
ScalarOrArray = TypeVar("ScalarOrArray", float, np.floating, NDArrayFloat)


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.weights: ScalarOrArray = np.random.uniform(-0.5, 0.5, (n_outputs, n_inputs))
        self.biases: ScalarOrArray = np.zeros((n_outputs, 1))


def sigmoid(x: ScalarOrArray | float) -> ScalarOrArray | float:
    return 1 / (1+np.exp(-x))


def sigmoid_derivative(y: ScalarOrArray | float) -> ScalarOrArray | float:
    return y * (1 - y)


def softmax(x: ScalarOrArray) -> ScalarOrArray:
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
    """Backpropagation.
    Modifies weights and biases of each layer based on the contribution to the error value.
    Returns modified layer objects."""
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
    """Calculates accuracy of the prediction."""
    correct = list(filter(
            lambda elem: elem[0] == np.argmax(elem[1]), zip(predictions, labels)
            ))
    return round(len(correct)/len(predictions)*100, n_decimals)


def run(inputs_batch: ScalarOrArray, labels_batch: ScalarOrArray,
        layers: list[Layer], epochs: int, learn_rate: float) -> list[int]:
    for _ in range(epochs):
        predictions = []
        for inputs, labels in zip(inputs_batch, labels_batch):
            # Convert vector to matrix
            labels.shape += (1,)

            # Make predictions
            layers_outputs = forward(layers, inputs)
            predictions.append(np.argmax(layers_outputs[-1]))

            # Train neural netrowk
            layers = backward(
                layers, layers_outputs, inputs, labels, learn_rate
                )
    return predictions


def show_random_bad_prediction(images, predictions, labels):
    bad_lst = []
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        if prediction != np.argmax(label):
            bad_lst.append((i, prediction, np.argmax(label)))

    bad = bad_lst[np.random.randint(len(bad_lst))]
    index = bad[0]
    prediction = bad[1]
    label = bad[2]
    img = images[index]

    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.title(f"Prediction: {prediction}, Label: {label}")
    plt.show()


def show_random_good_prediction(images, predictions, labels):
    good_lst = []
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        if prediction == np.argmax(label):
            good_lst.append((i, prediction, np.argmax(label)))

    good = good_lst[np.random.randint(len(good_lst))]
    index = good[0]
    prediction = good[1]
    label = good[2]
    img = images[index]

    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.title(f"Prediction: {prediction}, Label: {label}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        """Recognize hand-written digits, using MNIST data set."""
        )
    parser.add_argument('-n', action='append', type=int,
                        help='number of neurons in this hidden layer.',
                        required=True)
    parser.add_argument('--learn_rate', type=float, default=0.01,
                        help='learn rate of the neural network (default: 0.01).')
    parser.add_argument('--epochs', type=int, default=3,
                        help='epochs of the neural network (default: 3).')
    args = parser.parse_args()

    images, labels = get_mnist()

    # Create input layer and append it to list of layers
    input_layer = Layer(784, args.n[0])
    layers = [input_layer]
    # Create hidden layers and append them to list of layers
    if len(args.n) > 1:
        for i, _ in enumerate(args.n[:-1]):
            layers.append(Layer(args.n[i], args.n[i+1]))

    # Create output layer and append it to list of layers
    output_layer = Layer(args.n[-1], 10)
    layers.append(output_layer)

    predictions = run(images, labels, layers, args.epochs, args.learn_rate)
    print(calculate_accuracy(predictions, labels))