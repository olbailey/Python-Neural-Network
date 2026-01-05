import pandas as pd
import numpy as np
import os

class DataSet:
    '''Use DataSet().get_data() to access file data'''
    def __init__(self):
        pass

    def get_data(self, filename=None) -> list[np.ndarray]:
        '''returns values, labels'''
        if filename is None:
            files = os.listdir()
            csvfiles = []
            
            for filename in files:
                if filename[-4:] == '.csv':
                    csvfiles.append(filename)

            count = 1
            for csvfile in csvfiles:
                print(f"{count}. {csvfile}")
                count += 1
            choice = int(input('Enter file number: '))
        
            values, labels = self._convert(csvfiles[choice-1])
        else:
            values, labels = self._convert(filename)

        return values, labels

    def _convert(self, filename):
        df = pd.read_csv(filename)
        df_len = len(df.index)

        array_labels = np.empty(df_len, np.int64)
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