import numpy as np

# stevens' power law parameters
resolution = 100
proportionality_constant = 1.0
modality_constant = 0.8
maximum_stimulus_intensity = 5.0

# domain
x1_domain = np.linspace(1/resolution, maximum_stimulus_intensity, resolution)


def spl_experiment():
    X = []
    y = []
    for x1 in x1_domain:
        X.append([x1])
        y.append(spl(x1))
    return np.array(X), np.array(y)


def spl(x1: float):
    return proportionality_constant * x1 ** modality_constant
