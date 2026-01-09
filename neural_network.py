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

        self.learning_rate = learning_rate
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

        for i in range(self.size):
            self.layers.append(Layer(shape[i], shape[i + 1]))

        self.split_data()

    def train(self):
        stopped = False
        training_data_size = len(self.training_data)
        batches_num = training_data_size / self.batch_size

        NeuralNetwork.print_performance()

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
                NeuralNetwork.print_performance()


    def forward_pass(self, data_batch: np.ndarray) -> np.ndarray:
        batch_output = np.zeros((self.batch_size, self.layers[-1].nodes_num))

        for i in range(len(data_batch)):
            outputs = self.layers[0].calculate_output(data_batch[i], "Tanh")

            for j in range(1, self.size - 1):
                outputs = self.layers[j].calculate_output(outputs, "Tanh")

            batch_output[i] = self.layers[self.size - 1].calculate_output(outputs, "Softmax")

        return batch_output
    
    def loss(predicts: np.ndarray, labels: np.ndarray) -> float:
        corrected_labels = labels - 1

        losses = np.log(predicts[corrected_labels])
        losses *= -1

        return np.mean(losses)

    def backpropagation(self):
        pass

    def apply_gradients(self):
        pass
        
    def split_data(self):
        self.training_data = 
        self.training_labels = 
        self.testing_data = 
        self.testing_labels = 

    def print_performance():
        pass
        

if __name__ == "__main__":
    pass