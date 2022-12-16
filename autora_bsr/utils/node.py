import numpy as np
import pandas as pd
from typing import Union, Dict, Callable, Optional
from enum import Enum


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
    def __init__(
            self,
            depth: int = 0,
            node_type: NodeType = NodeType.EMPTY,
            left: "Node" = None,
            right: "Node" = None,
            parent: "Node" = None,
            operator: Callable = None,
            op_name: str = "",
            op_arity: int = 0
    ):
        # tree structure attributes
        self.depth = depth
        self.node_type = node_type
        self.left = left
        self.right = right
        self.parent = parent

        # a function that does the actual calculation, see definitions below
        self.operator = operator
        self.op_name = op_name
        self.op_arity = op_arity

        # holding temporary calculation result, see `evaluate()`
        self.result = None
        # params for additional inputs into `operator`
        self.params = {}

    def setup(
            self,
            op_name: str = "",
            operator: Optional[Callable] = None,
            op_arity: int = 0,
            **params: Dict
    ):
        self.op_name = op_name
        self.operator = operator
        self.op_arity = op_arity
        self.params = params

        if self.op_arity == 0:
            self.node_type = NodeType.LEAF
        elif self.op_arity == 1:
            self.left = Node(depth=self.depth + 1, parent=self)
            self.node_type = NodeType.UNARY
        elif self.op_arity == 2:
            self.left = Node(depth=self.depth + 1, parent=self)
            self.right = Node(depth=self.depth + 1, parent=self)
            self.node_type = NodeType.BINARY
        else:
            raise ValueError(
                "operation arity should be either 0, 1, 2; get {} instead".format(self._op_arity))

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], store_result: bool = False) -> np.array:
        if X is None:
            raise TypeError("input data X is non-existing")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if self.node_type == NodeType.LEAF:
            result = np.array(X.iloc[:, self.params["feature"]])
        elif self.node_type == NodeType.UNARY:
            result = self.operator(self.left.evaluate(X), **self.params)
        elif self.node_type == NodeType.BINARY:
            result = self.operator(self.left.evaluate(X), self.right.evaluate(X), **self.params)
        else:
            raise NotImplementedError("node evaluated before being setup")
        if store_result:
            self.result = result
        return result

    def __str__(self) -> str:
        raise NotImplementedError("TODO")
