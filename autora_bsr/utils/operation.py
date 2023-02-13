from typing import Callable, Dict

import numpy as np

"""
this file contains functions (operators) for actually carrying out the computations
in our expression tree model. An operator can take in either 1 (unary) or 2 (binary)
operands - corresponding to being used in a unary or binary node (see `node.py`). The
operand(s) are recursively evaluated `np.array` from an operation or literal (in the
case of a leaf node) in downstream node(s).

For certain operator, e.g. a linear operator, auxiliary parameters (slope/intercept)
are needed and can be passed in through `params` dictionary. These parameters are
initialized in `prior.py` by their specified initialization functions.
"""


# a linear operator with default `a` = 1 and `b` = 0 (i.e. identity operation)
def linear_op(operand: np.array, **params: Dict) -> np.array:
    a, b = params.get("a", 1), params.get("b", 0)
    return a * operand + b


# a safe `exp` operation that has a cutoff (default = 1e-10) and avoids overflow
def exp_op(operand: np.array, **params: Dict) -> np.array:
    cutoff = params.get("cutoff", 1e-10)
    return 1 / (cutoff + np.exp(-operand))


# a safe `inv` operation that has a cutoff (default = 1e-10) and avoids overflow
def inv_op(operand: np.array, **params: Dict) -> np.array:
    cutoff = params.get("cutoff", 1e-10)
    return 1 / (cutoff + operand)


def neg_op(operand: np.array) -> np.array:
    return -operand


def sin_op(operand: np.array) -> np.array:
    return np.sin(operand)


def cos_op(operand: np.array) -> np.array:
    return np.cos(operand)


# high-level func that produces power funcs such as `square`, `cubic`, etc.
def make_pow_op(power: int) -> Callable[[np.array], np.array]:
    def pow_op(operand: np.array) -> np.array:
        return np.power(operand, power)

    return pow_op


"""
a list of binary operators
"""


def plus_op(operand_a: np.array, operand_b: np.array):
    return operand_a + operand_b


def minus_op(operand_a: np.array, operand_b: np.array):
    return operand_a - operand_b


def multiply_op(operand_a: np.array, operand_b: np.array):
    return operand_a * operand_b
