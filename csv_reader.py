import numpy as np
import pandas as pd

class CsvReader:
    def get_data_circle_classification(file_name: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(file_name)
        labels = df["label"].to_numpy()
        labels -= 1
        data = df[["x", "y"]].to_numpy()

        data_squared = data ** 2
        extra_info = data_squared[:, 0] + data_squared[:, 1] # x^2 + y^2
        extra_info = np.sqrt(extra_info)

        data = np.column_stack((data, extra_info))
        return data, labels
    
    def get_mnist_data(training_file_name: str, testing_file_name: str) -> tuple[np.ndarray, np.ndarray]:
        df_training = pd.read_csv(training_file_name)
        df_testing = pd.read_csv(testing_file_name)

        training_data = df_training.iloc[:, 1:784 + 1].to_numpy()
        training_labels = df_training.iloc[:, 0].to_numpy()

        testing_data = df_testing.iloc[:, 1:784 + 1].to_numpy()
        testing_labels = df_testing.iloc[:, 0].to_numpy()

        data = np.concatenate((training_data, testing_data), axis=0)
        labels = np.concatenate((training_labels, testing_labels), axis=0)

        return data, labels

if __name__ == "__main__":
    x, y = CsvReader.get_mnist_data("data/mnist_train.csv", "data/mnist_test.csv")
    print(x.shape, y.shape)