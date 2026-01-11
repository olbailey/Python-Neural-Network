from csv_reader import CsvReader
from neural_network import NeuralNetwork 

BATCH_SIZE = 64

data, labels = CsvReader.get_data("data/centralCircleSmall.csv")

boby = NeuralNetwork(data, labels, (2, 16, 2), 0.0003, BATCH_SIZE)
boby.train()
