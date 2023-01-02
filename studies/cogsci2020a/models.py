import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

ground_truth_resolution = 10

def weber_fechner_data(metadata):

    s1_values = metadata.independent_variables[0].allowed_values
    s2_values = metadata.independent_variables[1].allowed_values

    X = np.array(np.meshgrid(s1_values, s2_values)).T.reshape(-1,2)
    # remove all combinations where s1 > s2
    X = X[X[:,0] <= X[:,1]]

    y = weber_fechner_experiment(X, std=0)

    return X, y

def weber_fechner_experiment(X: np.ndarray, weber_constant: float = 1.0, std = 0.1):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):
        # jnd =  np.min(x) * weber_constant
        # response = (x[1]-x[0]) - jnd
        # y = 1/(1+np.exp(-response)) + np.random.normal(0, std)
        y = weber_constant * np.log(x[1]/x[0]) + np.random.normal(0, std)
        Y[idx] = y

    return Y

def weber_fechner_metadata():
    iv1 = IV(
        name="S1",
        allowed_values=np.linspace(1/ground_truth_resolution, 5, ground_truth_resolution),
        value_range=(1, 5),
        units="intensity",
        variable_label="Stimulus 1 Intensity",
    )

    iv2 = IV(
        name="S2",
        allowed_values=np.linspace(1/ground_truth_resolution, 5, ground_truth_resolution),
        value_range=(1, 5),
        units="intensity",
        variable_label="Stimulus 2 Intensity",
    )

    dv1 = DV(
        name="difference_detected",
        value_range=(0, 1),
        units="sensation",
        variable_label="Sensation",
        type=ValueType.REAL,
    )

    metadata = VariableCollection(
        independent_variables=[iv1, iv2],
        dependent_variables=[dv1],
    )

    return metadata

model_inventory = dict()
model_inventory["weber_fechner"] = (weber_fechner_metadata(),
                                    weber_fechner_data(weber_fechner_metadata()),
                                    weber_fechner_experiment)


