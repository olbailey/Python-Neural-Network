import math

import numpy as np
from layer import Layer

class NeuralNetwork:
    BETA = 0.95
    TRAINING_SPLIT = 0.8

    def __init__(self, data: np.ndarray, labels: np.ndarray, shape: tuple, learning_rate: float, batch_size: int):
        self.iterations = 0

        self.layers: list[Layer] = []
        self.size = len(shape) - 1
        self.N = len(data)

        # Shuffling data
        random_indexes = np.arange(self.N)
        np.random.shuffle(random_indexes)
        self.data = data[random_indexes]
        self.labels = labels[random_indexes]

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Creating network layers
        for i in range(self.size):
            self.layers.append(Layer(shape[i], shape[i + 1], batch_size))

        # Splitting training and testing data
        testing_cut_off = math.floor(self.N * self.TRAINING_SPLIT)

        self.training_data = self.data[0 : testing_cut_off]
        self.training_labels = self.labels[0 : testing_cut_off]

        self.testing_data = self.data[testing_cut_off : self.N]
        self.testing_labels = self.labels[testing_cut_off : self.N]

    def train(self) -> None:
        stopped = False
        training_data_size = len(self.training_data)
        batches_num = round(training_data_size / self.batch_size)

        self.print_performance()

        while not stopped:
            for i in range(1, batches_num):
                start_point = self.batch_size * (i - 1)
                end_point = self.batch_size * i

                if end_point > training_data_size:
                    end_point = training_data_size
                
                training_batch = self.training_data[start_point : end_point]
                labels_batch = self.training_labels[start_point : end_point]

                self.forward_pass(training_batch)
                self.backpropagation(training_batch, labels_batch)
                self.apply_gradients()
            
            self.iterations += 1

            if (self.iterations % 1000 == 0):
                self.print_performance()


    def forward_pass(self, data_batch: np.ndarray) -> np.ndarray:
        outputs = self.layers[0].calculate_output(data_batch, "Tanh")

        for j in range(1, self.size - 1):
            outputs = self.layers[j].calculate_output(outputs, "Tanh")

        return self.layers[self.size - 1].calculate_output(outputs, "Softmax")
    
    def loss(predicts: np.ndarray, labels: np.ndarray) -> float:
        corrected_labels = labels - 1

        losses = np.log(predicts[corrected_labels])
        losses *= -1

        return np.mean(losses)

    def backpropagation(self, training_data: np.ndarray, training_labels: np.ndarray):
        self.layers[-1].backpropagation_output_layer(self.layers[self.size - 2], training_labels)

                # int i = size - 2; i >=0; --i
        for i in range(self.size - 2, -1, -1):
            self.layers[i].backpropagation_hidden_layer(self.layers[i + 1], training_data)

    def apply_gradients(self) -> None:
        for layer in self.layers:
            layer.apply_gradients(self.learning_rate, self.BETA)

    def cost(predicts: np.ndarray, labels: np.ndarray) -> int:
        predicted_outputs = np.argmax(predicts, axis=1)
        predicted_outputs += 1

        return np.sum(np.equal(predicted_outputs, labels, out=predicted_outputs))

    def print_performance(self) -> None:
        testing_predicts = self.forward_pass(self.testing_data)
        
        print(f"Iterations: {self.iterations}")
        print(f"Testing data loss: {NeuralNetwork.loss(testing_predicts, self.testing_labels)}")
        print(f"Accuracy: {NeuralNetwork.cost(testing_predicts, self.testing_labels)} / {self.testing_labels.size} \n")
        

if __name__ == "__main__":
    pass