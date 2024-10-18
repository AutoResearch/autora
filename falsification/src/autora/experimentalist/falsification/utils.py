from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from autora.variable import IV, VariableCollection


def plot_falsification_diagnostics(
    losses,
    popper_input,
    popper_input_full,
    popper_prediction,
    popper_target,
    model_prediction,
    target,
):
    import matplotlib.pyplot as plt

    if popper_input.shape[1] > 1:
        plot_input = popper_input[:, 0]
    else:
        plot_input = popper_input

    if model_prediction.ndim > 1:
        if model_prediction.shape[1] > 1:
            model_prediction = model_prediction[:, 0]
            target = target[:, 0]

    # PREDICTED MODEL ERROR PLOT
    plot_input_order = np.argsort(np.array(plot_input).flatten())
    plot_input = plot_input[plot_input_order]
    popper_target = popper_target[plot_input_order]
    # popper_prediction = popper_prediction[plot_input_order]
    plt.plot(
        popper_input_full,
        popper_prediction.detach().numpy(),
        label="Predicted MSE of the Model",
    )
    plt.scatter(
        plot_input,
        popper_target.detach().numpy(),
        s=20,
        c="red",
        label="True MSE of the Model",
    )
    plt.xlabel("Experimental Condition X")
    plt.ylabel("MSE of Model")
    plt.title("Prediction of Falsification Network")
    plt.legend()
    plt.show()

    # CONVERGENCE PLOT
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss for the Falsification Network")
    plt.show()

    # MODEL PREDICTION PLOT
    model_prediction = model_prediction[plot_input_order]
    target = target[plot_input_order]
    plt.plot(plot_input, model_prediction, label="Model Prediction")
    plt.scatter(plot_input, target, s=20, c="red", label="Data")
    plt.xlabel("Experimental Condition X")
    plt.ylabel("Observation Y")
    plt.title("Model Prediction Vs. Data")
    plt.legend()
    plt.show()


def class_to_onehot(y: np.array, n_classes: Optional[int] = None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        n_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not n_classes:
        n_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, n_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (n_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_iv_limits(
    reference_conditions: np.ndarray,
    metadata: VariableCollection,
):
    """
    Get the limits of the independent variables

    Args:
        reference_conditions: data that the model was trained on
        metadata: Meta-data about the dependent and independent variables

    Returns: List of limits for each independent variable
    """

    # create list of IV limits
    iv_limit_list = list()
    if metadata is not None:
        ivs = metadata.independent_variables
        for iv in ivs:
            if hasattr(iv, "value_range"):
                value_range = cast(Tuple, iv.value_range)
                lower_bound = value_range[0]
                upper_bound = value_range[1]
                iv_limit_list.append(([lower_bound, upper_bound]))
    else:
        for col in range(reference_conditions.shape[1]):
            min = np.min(reference_conditions[:, col])
            max = np.max(reference_conditions[:, col])
            iv_limit_list.append(([min, max]))

    return iv_limit_list


def align_dataframe_to_ivs(
    dataframe: pd.DataFrame, independent_variables: List[IV]
) -> pd.DataFrame:
    """
    Aligns a dataframe to a metadata object, ensuring that the columns are in the same order
    as the independent variables in the metadata.

    Args:
        dataframe: a dataframe with columns to align
        independent_variables: a list of independent variables

    Returns:
        a dataframe with columns in the same order as the independent variables in the metadata
    """
    variable_names = list()
    for variable in independent_variables:
        variable_names.append(variable.name)
    return dataframe[variable_names]
