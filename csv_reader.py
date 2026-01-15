import numpy as np
import pandas as pd

class CsvReader:
    def get_data(file_name: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(file_name)
        labels = np.array(df["label"])
        data = np.array(df[["x", "y"]])

        data_squared = data ** 2
        extra_info = data_squared[:, 0] + data_squared[:, 1] # x^2 + y^2
        extra_info = np.sqrt(extra_info)

        data = np.column_stack((data, extra_info))
        return data, labels

if __name__ == "__main__":
    x, y = CsvReader.get_data("data/centralCircleSmall.csv")
    print(x[54])