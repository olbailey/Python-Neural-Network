import matplotlib.pyplot as plt
import numpy as np

def display(predictions: np.ndarray, points: np.ndarray = None): # 
    if points is None:
        points = np.array([[x, y] for y in range(0, 187) for x in range(0, 187)])

    fig, ax = plt.subplots()
    xs = points[:, 0]
    ys = points[:, 1]

    scatter = ax.scatter(xs, ys, c=predictions, cmap='bwr', vmin=0, vmax=1, s=1)
    plt.show()
    

if __name__ == "__main__":
    display(np.ones(34969))