import numpy as np
import pandas as pd
from typing import Union, Dict, Callable, Optional
from enum import Enum
from funcs import get_expression


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
            op_arity: int = 0,
            op_init: Callable = None,
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
        self.op_init = op_init

        # holding temporary calculation result, see `evaluate()`
        self.result = None
        # params for additional inputs into `operator`
        self.params = {}

    def _init_param(self, **hyper_params):
        # init is a function randomized by some hyper-params
        if callable(self.op_init):
            self.params = self.op_init(**hyper_params)
        else:  # init is deterministic dict
            self.params = self.op_init

    def setup(
            self,
            op_name: str = "",
            ops_prior: Dict = {},
            feature: int = 0,
            **hyper_params
    ):
        """
        Initialize an uninitialized node with given feature, in the case of a leaf node, or some
        given operator information, in the case of unary or binary node. The type of the node is
        determined by the feature/operator assigned to it.

        Arguments:
            op_name: the operator name, if given
            ops_prior: the prior dictionary of the given operator
            feature: the index of the assigned feature, if given
            hyper_params: hyperparameters for initializing the node
        """
        self.op_name = op_name
        self.operator = ops_prior.get("fn")
        self.op_arity = ops_prior.get("arity", 0)
        self.op_init = ops_prior.get("init", {})
        self._init_param(**hyper_params)

        if self.op_arity == 0:
            self.params["feature"] = feature
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
                "operation arity should be either 0, 1, 2; get {} instead".format(self.op_arity))

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], store_result: bool = False) -> np.array:
        """
        Evaluate the expression, as represented by an expression tree with `self` as the root,
        using the given data matrix `X`.

        Arguments:
            X: the data matrix with each row being a data point and each column a feature
            store_result: whether to store the result of this calculation

        Return:
            result: the result of this calculation
        """
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
        """
        Get a literal (string) representation of a tree `node` data structure.
        See `get_expression` for more information.
        """
        return get_expression(self)
