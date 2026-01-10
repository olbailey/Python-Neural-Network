import numpy as np
import pandas as pd

class CsvReader:
    def get_data(file_name: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(file_name)
        data = np.array(df[["x", "y"]])
        labels = np.array(df["label"])
        return data, labels

if __name__ == "__main__":
    CsvReader.get_data("data/centralCircleSmall.csv")