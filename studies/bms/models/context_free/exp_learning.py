import numpy as np

resolution = 100
minimum = 0
maximum = 0.5
lr = 0.03
p_asymptotic = 1.0

trials = range(resolution)
initial_values = np.linspace(minimum, maximum, resolution)


def exp_learning_experiment():
    X = []
    y = []
    for x1 in trials:
        for x2 in initial_values:
            X.append([x1, x2])
            y.append(exp_learning(x1, x2))
    return np.array(X), np.array(y)


def exp_learning(x1: float, x2: float):
    p_initial = x1
    trial = x2
    y = p_asymptotic - (p_asymptotic - p_initial) * np.exp(- lr * trial)
    return y
