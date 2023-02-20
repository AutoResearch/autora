from typing import List, Union

import numpy as np

from autora.theorist.bsr.node import Node
from autora.theorist.bsr.prior import get_prior_dict


def _build_tree_from_literals(literals: List[Union[str, int]], **hyper_params):
    """
    Helper testing function that builds up a valid computation tree with a list of str/int inputs
    where a string input represents an operation (e.g. `inv`, `+`) and an integer indicates which
    feature to use in a leaf node. For the list of valid operations, see `priors.py`.

    The construction is done level-by-level. For example, the list `["sin", "inv", 1, 0] will render
    the following computation tree

                sin (root)
                /       \
             inv      feature 1
             /
        feature 0

    Note that for simplicity this function doesn't check the validity of input list; e.g. using
    a binary operation without specifying the features used by its two leaf nodes might cause error.

    Arguments:
        literals: a list of strings and integers that specifies how the tree should be built
        hyper_params: parameters to initialize certain operations
    Returns:
        root: the root node of the tree
    """
    _, _, prior_dict = get_prior_dict()
    root = Node(0)
    node_queue = [root]
    for s in literals:
        node = node_queue.pop(0)
        params = {}
        if isinstance(s, str):
            ops_init = prior_dict[s]["init"]
            # init is a function randomized by some hyper-params
            if callable(ops_init):
                params.update(ops_init(**hyper_params))
            else:  # init is deterministic dict
                params.update(ops_init)
            node.setup(s, prior_dict[s])
        elif isinstance(s, int):
            params["feature"] = s
            node.setup(**params)
        if node.left:
            node_queue.append(node.left)
        if node.right:
            node_queue.append(node.right)
    return root


def test_basic_linear_operation():
    root = _build_tree_from_literals(["ln", 0])
    root.params.update({"a": 2, "b": 3})
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = 2 * test_x[:, 0] + 3
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_exp_operation():
    root = _build_tree_from_literals(["exp", 0])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = np.exp(test_x[:, 0])
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_inv_operation():
    root = _build_tree_from_literals(["inv", 0])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = 1 / test_x[:, 0]
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_neg_operation():
    root = _build_tree_from_literals(["neg", 0])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = -test_x[:, 0]
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_sin_operation():
    root = _build_tree_from_literals(["sin", 0])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = np.sin(test_x[:, 0])
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_cos_operation():
    root = _build_tree_from_literals(["cos", 0])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = np.cos(test_x[:, 0])
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_plus_operation():
    root = _build_tree_from_literals(["+", 0, 1])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = np.add(test_x[:, 0], test_x[:, 1])
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_minus_operation():
    root = _build_tree_from_literals(["-", 0, 1])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = np.subtract(test_x[:, 0], test_x[:, 1])
    assert (test_y - root.evaluate(test_x) < 1e-5).all()


def test_basic_multiply_operation():
    root = _build_tree_from_literals(["*", 0, 1])
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = np.multiply(test_x[:, 0], test_x[:, 1])
    assert (test_y - root.evaluate(test_x) < 1e-5).all()
