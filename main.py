from csv_reader import CsvReader
from neural_network import NeuralNetwork 

BATCH_SIZE = 64

data, labels = CsvReader.get_data("data/centralCircleLarge.csv")

boby = NeuralNetwork(data, labels, (3, 64, 2), 0.0003, BATCH_SIZE)
boby.train()
