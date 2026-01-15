from typing import Self
import math

import numpy as np

class Layer:
    ACTIVATION_FUNCTIONS = ["Tanh", "ReLu"]
    ACT_FUNC = ACTIVATION_FUNCTIONS[1]
    OUT_FUNC = "Softmax"

    def __init__(self, nodes_in: int, nodes_num: int) -> None:
        self.nodes_in = nodes_in
        self.nodes_num = nodes_num

        self.weights = Layer.initialise_weights(nodes_num, nodes_in)
        self.biases = np.zeros(nodes_num)

        self.costGradientWeights = np.zeros((nodes_num, nodes_in))
        self.costGradientBiases = np.zeros(nodes_num)

        self.batch_outputs: np.ndarray = None
        self.batch_error_signals: np.ndarray = None

        self.velocityWeights = np.zeros((nodes_num, nodes_in))
        self.velocityBiases = np.zeros(nodes_num)

    def calculate_output(self, inputs_batch: np.ndarray, Hidden_activation: bool) -> np.ndarray:
        layer_outputs = np.zeros(self.nodes_num)

        layer_outputs = (self.weights @ inputs_batch.T).T + self.biases

        if Hidden_activation:
            if (Layer.ACT_FUNC == "Tanh"):
                np.tanh(layer_outputs, out=layer_outputs)

            elif (Layer.ACT_FUNC == "ReLu"):
                layer_outputs[layer_outputs < 0] = 0
        else:
            if (Layer.OUT_FUNC == "Softmax"):
                Layer.softmax(layer_outputs)

        self.batch_outputs = layer_outputs.copy()
        return layer_outputs
    
    def apply_gradients(self, learning_rate: float, beta: float) -> None:
        self.velocityBiases *= beta
        self.costGradientBiases *= (1 - beta)
        self.velocityBiases += self.costGradientBiases

        self.biases -= self.velocityBiases * learning_rate

        self.velocityWeights *= beta
        self.costGradientWeights *= (1 - beta)
        self.velocityWeights += self.costGradientWeights

        self.weights -= self.velocityWeights * learning_rate

        self.costGradientWeights.fill(0)
        self.costGradientBiases.fill(0)

    def backpropagation_output_layer(self, previous_layer: Self, training_labels: np.ndarray) -> None:
        batch_size = training_labels.shape[0]
        corrected_labels = training_labels - 1

        self.batch_error_signals = self.hot_vector_output(corrected_labels)

        self.costGradientBiases += np.sum(self.batch_error_signals, axis=0)
        self.costGradientWeights += self.batch_error_signals.T @ previous_layer.batch_outputs

        self.costGradientBiases /= batch_size
        self.costGradientWeights /= batch_size

    def backpropagation_hidden_layer(self, previous_layer: Self, input_data: np.ndarray) -> None:
        batch_size = input_data.shape[0]
        
        self.batch_error_signals = (previous_layer.weights.T @ previous_layer.batch_error_signals.T).T

        self.batch_error_signals *= Layer.calculate_derivative(self.batch_outputs)

        self.costGradientBiases += np.sum(self.batch_error_signals, axis=0)
        self.costGradientWeights += self.batch_error_signals.T @ input_data

        self.costGradientBiases /= batch_size
        self.costGradientWeights /= batch_size          

    def softmax(inputs: np.ndarray) -> None:
        max_values = np.max(inputs, axis=1)

        inputs -= max_values[:, None]
        np.exp(inputs, out=inputs)

        norm_base = np.sum(inputs, axis=1)

        inputs /= norm_base[:, None]

    def xavier(fan_out, fan_in):
        limit = math.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_out, fan_in))
    
    def he_uniform(fan_out, fan_in):
        limit = math.sqrt(6 / (fan_in))

        return np.random.uniform(-limit, limit, size=(fan_out, fan_in))

    def initialise_weights(nodes_num, nodes_in) -> np.ndarray:
        if Layer.ACT_FUNC == "Tanh":
            return Layer.xavier(nodes_num, nodes_in)
        elif Layer.ACT_FUNC == "ReLu":
            return Layer.he_uniform(nodes_num, nodes_in)

    def hot_vector_output(self, labels: np.ndarray) -> np.ndarray:
        values = self.batch_outputs.copy()
        values[np.arange(values.shape[0]), labels] -= 1
        return values

    def calculate_derivative(inputs: np.ndarray):
        if Layer.ACT_FUNC == "Tanh":
            return  1 - (inputs ** 2)
        elif Layer.ACT_FUNC == "ReLu":
            values = inputs.copy()
            values[values > 0] = 1
            values[values < 0] = 0
            return values


if __name__ == "__main__":
    pass
