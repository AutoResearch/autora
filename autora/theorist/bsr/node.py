from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .misc import get_ops_expr


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
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        parent: Optional["Node"] = None,
        operator: Optional[Callable] = None,
        op_name: str = "",
        op_arity: int = 0,
        op_init: Optional[Callable] = None,
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
        self.params: Dict = {}

    def _init_param(self, **hyper_params):
        # init is a function randomized by some hyper-params
        if callable(self.op_init):
            self.params = self.op_init(**hyper_params)
        else:  # init is deterministic dict
            self.params = self.op_init

    def setup(
        self, op_name: str = "", ops_prior: Dict = {}, feature: int = 0, **hyper_params
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
        self.operator = ops_prior.get("fn", None)
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
                "operation arity should be either 0, 1, 2; get {} instead".format(
                    self.op_arity
                )
            )

    def evaluate(
        self, X: Union[np.ndarray, pd.DataFrame], store_result: bool = False
    ) -> np.array:
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
            result = np.array(X.iloc[:, self.params["feature"]]).flatten()
        elif self.node_type == NodeType.UNARY:
            assert self.left and self.operator
            result = self.operator(self.left.evaluate(X), **self.params)
        elif self.node_type == NodeType.BINARY:
            assert self.left and self.right and self.operator
            result = self.operator(
                self.left.evaluate(X), self.right.evaluate(X), **self.params
            )
        else:
            raise NotImplementedError("node evaluated before being setup")
        if store_result:
            self.result = result
        return result

    def get_expression(
        self,
        ops_expr: Optional[Dict[str, str]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Get a literal (string) expression of the expression tree

        Arguments:
            ops_expr: the dictionary that maps an operation name to its literal format; if not
                offered, use the default one in `get_ops_expr()`
            feature_names: the list of names for the data features
        Return:
            a literal expression of the tree
        """
        if not ops_expr:
            ops_expr = get_ops_expr()
        if self.node_type == NodeType.LEAF:
            if feature_names:
                return feature_names[self.params["feature"]]
            else:
                return f"x{self.params['feature']}"
        elif self.node_type == NodeType.UNARY:
            # if the expr for an operator is not defined, use placeholder
            # e.g. operator `cosh` -> `cosh(xxx)`
            assert self.left
            place_holder = self.op_name + "({})"
            left_expr = self.left.get_expression(ops_expr, feature_names)
            expr_fmt = ops_expr.get(self.op_name, place_holder)
            return expr_fmt.format(left_expr, **self.params)
        elif self.node_type == NodeType.BINARY:
            assert self.left and self.right
            place_holder = self.op_name + "({})"
            left_expr = self.left.get_expression(ops_expr, feature_names)
            right_expr = self.right.get_expression(ops_expr, feature_names)
            expr_fmt = ops_expr.get(self.op_name, place_holder)
            return expr_fmt.format(left_expr, right_expr, **self.params)
        else:  # empty node
            return "(empty node)"

    def __str__(self) -> str:
        """
        Get a literal (string) representation of a tree `node` data structure.
        See `get_expression` for more information.
        """
        return self.get_expression()
