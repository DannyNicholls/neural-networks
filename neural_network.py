import numpy as np


def activation(z, function):
    if function == 'relu':
        return np.fmax(0, z)
    if function == 'sigmoid':
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    array = np.array([-100, 0, 100])
    print(activation(array, 'relu'))
