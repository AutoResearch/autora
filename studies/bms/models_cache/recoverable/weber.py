import numpy as np
from numpy import log

x1_domain = range(1, 10)
x2_domain = range(1, 10)


def weber_experiment():
    X = []
    y = []
    for x1 in x1_domain:
        for x2 in x2_domain:
            if x2 >= x1:
                X.append([x1, x2])
                y.append(weber(x1, x2))
    return np.array(X), np.array(y)


def weber(x1: float, x2: float):
    return log(x1 / x2)
