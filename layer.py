import numpy as np

class Layer:
    def __init__(self, nodes_in: int, nodes_num: int, batch_size: int):
        self.nodes_in = nodes_in
        self.nodes_num = nodes_num

        self.weights = Layer.initialise_weights(nodes_num, nodes_in)
        self.biases = np.zeros(nodes_num)

        self.costGradientWeights = np.zeros((nodes_in, nodes_num))
        self.costGradientBiases = np.zeros(nodes_num)

        self.batch_outputs = np.zeros((batch_size, nodes_num))
        self.batch_error_signals = np.zeros((batch_size, nodes_num))

        self.velocityWeights = np.zeros((nodes_num, nodes_in))
        self.velocityBiases = np.zeros(nodes_num)

    def calculate_output(self, inputs: np.ndarray, activation_name: str) -> np.ndarray:
        layer_outputs = np.zeros(self.nodes_num)

        layer_outputs = np.dot(self.weights, inputs) + self.biases

        if (activation_name == "Tanh"):
            np.tanh(layer_outputs, out=layer_outputs)
        elif (activation_name == "Softmax"):
            Layer.softmax(layer_outputs)

        self.batch_outputs.append(layer_outputs)
        return layer_outputs

    def softmax(inputs: np.ndarray):
        max_value = np.max(inputs)

        inputs -= max_value
        np.exp(inputs, out=inputs)

        norm_base = np.sum(inputs)

        inputs /= norm_base


    def xavier():
        pass

    def initialise_weights(nodes_in, nodes_num) -> np.ndarray:
        pass

if __name__ == "__main__":
    pass
