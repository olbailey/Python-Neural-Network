from csv_reader import CsvReader
from neural_network import NeuralNetwork 

BATCH_SIZE = 64

# Circle Classification - loss = 0.02 at 11000 iterations
# data, labels = CsvReader.get_data_circle_classification("data/centralCircleLarge.csv")
# boby = NeuralNetwork(data, labels, (3, 64, 2), 0.0003, BATCH_SIZE)

# MNIST Classification - loss = 0.42 after 70 iterations
data, labels = CsvReader.get_mnist_data("data/mnist_train.csv", "data/mnist_test.csv")

boby = NeuralNetwork(data, labels, (784, 256, 128, 10), 0.0003, BATCH_SIZE)

boby.train()
