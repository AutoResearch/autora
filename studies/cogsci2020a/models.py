import numpy as np
from autora.experimentalist.filter import weber_filter
from autora.variable import DV, IV, ValueType, VariableCollection

def weber_fechner_experiment(X: np.ndarray, weber_constant: float = 0.5):
    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):
        jnd =  np.min(x) * weber_constant
        response = (x[1]-x[0]) - jnd
        y = 1/(1+np.exp(-response))
        Y[idx] = y

    return Y

def weber_fechner_metadata():
    iv1 = IV(
        name="S1",
        allowed_values=np.linspace(0, 5, 5),
        units="intensity",
        variable_label="Stimulus 1 Intensity",
    )

    iv2 = IV(
        name="S2",
        allowed_values=np.linspace(0, 5, 5),
        units="intensity",
        variable_label="Stimulus 2 Intensity",
    )

    # The experimentalist pipeline doesn't actually use DVs, they are just specified here for
    # example.
    dv1 = DV(
        name="difference_detected",
        value_range=(0, 1),
        units="probability",
        variable_label="P(difference detected)",
        type=ValueType.PROBABILITY,
    )

    # Variable collection with ivs and dvs
    metadata = VariableCollection(
        independent_variables=[iv1, iv2],
        dependent_variables=[dv1],
    )

    return metadata

model_inventory = dict()
model_inventory["weber_fechner"] = (weber_fechner_metadata, weber_filter, weber_fechner_experiment)
