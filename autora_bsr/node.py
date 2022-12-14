import numpy as np
import pandas as pd
from typing import Union, Dict, Callable
from enum import Enum
from abc import ABC, abstractmethod


class NodeType(Enum):
    """
    -1 represents newly grown node (not decided yet)
    0 represents no child, as a terminal node
    1 represents one child,
    2 represents 2 children
    """
    EMPTY = -1
    LEAF = 0
    UNARY = 1
    BINARY = 2


class Node:
    def __init__(self, depth: int = 0):
        # tree structure attributes
        self.type: NodeType = NodeType.EMPTY
        self.left: Node = None
        self.right: Node = None
        self.depth = depth
        self.parent = None

        # calculation attributes
        self.operator = None
        # operator is a string, either "+","*","ln","exp","inv"
        self.result = None
        # feature is a int indicating the index of feature in the input data
        self.feature = None

        # params for additional
        self.params = {}

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], store_result: bool = False) -> np.array:
        if X is None:
            raise TypeError("input data X is non-existing")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if self.type == NodeType.LEAF:
            result = np.array(X.iloc[:, self.feature])
        elif self.type == NodeType.UNARY:
            result = self.operator(self.left.evaluate(X), **self.params)
        elif self.type == NodeType.BINARY:
            result = self.operator(self.left.evaluate(X), self.right.evaluate(X), **self.params)
        else:
            raise NotImplementedError("node evaluated before being fully grown")
        if store_result:
            self.result = result
        return result

    def __str__(self) -> str:
        raise NotImplementedError("TODO")


"""
a list of unary operators
"""


# a linear operator with default `a` = 1 and `b` = 0 (i.e. identity operation)
def ln_op(operand: np.array, **params: Dict) -> np.array:
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
