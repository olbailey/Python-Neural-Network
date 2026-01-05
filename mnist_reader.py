import pandas as pd
import numpy as np

class DataSet:
    '''Use DataSet().get_data() to access file data'''
    def __init__(self):
        pass

    def get_data(self) -> list[np.ndarray]:
        return self._combine()

    def _combine(self):
        # 'Mnist_ai\mnist_train.csv'
        # 'Mnist_ai\mnist_test.csv'
        values_1, labels_1 = self._convert('Mnist_ai/mnist_train.csv')
        values_2, labels_2 = self._convert('Mnist_ai/mnist_test.csv')

        values = np.append(values_1, values_2)
        labels = np.append(labels_1, labels_2)

        return values, labels

    def _convert(self, filename):
        df = pd.read_csv(filename)
        df_len = len(df.index)

        array_labels = np.empty(df_len, np.ndarray)
        array_values = np.empty(df_len, np.ndarray)

        for i in range(df_len):
            data = np.array(df.iloc[i])
            array_labels[i] = np.array(data[0])
            array_values[i] = np.array(data[1:])

        return array_values, array_labels

if __name__ == '__main__':
    x = DataSet().get_data()
    print('#'*10)
    print(x[0].size)