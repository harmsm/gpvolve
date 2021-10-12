import numpy as np


def sigmoidal(P, K, n, a=0, b=1):
    """
    """

    x = np.power(P, n)
    y = np.power(K, n)

    F = a * (x / (x + y)) + b

    if np.sum(np.isnan(F)) != 0 or \
            np.sum(np.isinf(F)) != 0 or \
            np.sum(F < 0) != 0:
        print(yo)
