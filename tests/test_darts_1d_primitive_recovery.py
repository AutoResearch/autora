#!/usr/bin/env python
from typing import Callable, Sequence

import numpy as np
from skl.darts import DARTSRegressor

primitives = ["none", "add", "subtract", "relu", "sigmoid", "mult", "exp", "1/x", "ln"]


def generate_x(start=-1, stop=1, num=100):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def transform_through_primitive_none(x: np.ndarray) -> np.ndarray:
    return x * 0.0


def transform_through_primitive_add(x: np.ndarray) -> np.ndarray:
    return x


def transform_through_primitive_subtract(x: np.ndarray) -> np.ndarray:
    return -x


def transform_through_primitive_relu(x: np.ndarray):
    y = x.copy()
    y[x < 0.0] = 0.0
    return y


def get_primitive_from_single_node_model(model):
    primitive = model[0].primitives[np.argmax(model[0].max_alphas_normal()).numpy()]
    return primitive


def run_test_primitive_fitting(
    X: np.ndarray,
    transformer: Callable,
    expected_primitive: str,
    primitives: Sequence[str] = primitives,
):
    y = transformer(X)
    regressor = DARTSRegressor(
        num_graph_nodes=1,
        param_updates_per_epoch=100,
        max_epochs=100,
        arch_updates_per_epoch=100,
        primitives=primitives,
    )
    regressor.fit(X, y)
    print("\n", np.column_stack((X, y, regressor.predict(X))).round(2))
    assert get_primitive_from_single_node_model(regressor.model_) == expected_primitive


def test_primitive_fitting_restricted_none():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_none,
        "none",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_restricted_add():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_add,
        "add",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_restricted_subtract():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_subtract,
        "subtract",
        primitives=["none", "add", "subtract"],
    )


def test_primitive_fitting_none():
    run_test_primitive_fitting(
        generate_x(),
        transform_through_primitive_none,
        "none",
        primitives=[
            "none",
            "add",
            "subtract",
            "relu",
            "sigmoid",
            "mult",
            "exp",
            "1/x",
            "ln",
        ],
    )


def test_primitive_fitting_lin_relu():
    run_test_primitive_fitting(
        generate_x(), transform_through_primitive_relu, "lin_relu"
    )


def test_primitive_fitting_add():
    run_test_primitive_fitting(generate_x(), transform_through_primitive_add, "add")


def test_primitive_fitting_subtract():
    run_test_primitive_fitting(
        generate_x(), transform_through_primitive_subtract, "subtract"
    )
