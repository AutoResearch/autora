from typing import List, Union

from autora.theorist.bsr.funcs import (
    de_transform,
    get_all_nodes,
    grow,
    prune,
    reassign_feat,
    reassign_op,
    transform,
)
from autora.theorist.bsr.node import Node, NodeType, Optional
from autora.theorist.bsr.prior import get_prior_dict


def __build_tree_from_literals(literals: List[Union[str, int]], **hyper_params):
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


def _assert_tree_completeness(
    tree: Node, depth: int = 0, parent: Optional[Node] = None
):
    assert tree.depth == depth
    assert tree.node_type != NodeType.EMPTY
    if parent:
        assert tree.parent is parent
    if tree.node_type == NodeType.LEAF:
        assert tree.op_name == ""
        assert "feature" in tree.params
    elif tree.node_type == NodeType.UNARY:
        assert tree.op_arity == 1
        assert tree.left and not tree.right
        _assert_tree_completeness(tree.left, depth + 1, tree)
    else:
        assert tree.op_arity == 2 and tree.left and tree.right
        if tree.op_init:  # operation with params
            assert len(tree.params) > 0
        _assert_tree_completeness(tree.left, depth + 1, tree)
        _assert_tree_completeness(tree.right, depth + 1, tree)


def test_mcmc_grow():
    ops_name_list, ops_weight_list, ops_priors = get_prior_dict()
    hyper_params = {"sigma_a": 1, "sigma_b": 1}
    node = __build_tree_from_literals(["*", "+", "-", 0, 1, 0, 1], **hyper_params)
    grow(node.left.left, ops_name_list, ops_weight_list, ops_priors, **hyper_params)
    _assert_tree_completeness(node)

    node = __build_tree_from_literals([0], **hyper_params)
    grow(node, ops_name_list, ops_weight_list, ops_priors, **hyper_params)
    _assert_tree_completeness(node)
    assert len(get_all_nodes(node)) > 1


def test_mcmc_prune():
    hyper_params = {"sigma_a": 1, "sigma_b": 1}
    node = __build_tree_from_literals(["*", "+", "-", 0, 1, 0, 1], **hyper_params)
    prune(node.right)
    _assert_tree_completeness(node)
    assert node.left.op_name == "+"
    assert node.right.node_type == NodeType.LEAF


def test_mcmc_de_transform(**hyper_params):
    hyper_params = {"sigma_a": 1, "sigma_b": 1}
    node = __build_tree_from_literals(["*", "exp", "-", 0, 0, 1], **hyper_params)
    repl, disc = de_transform(node.left)  # the unary case, replaced with child
    assert disc is None
    assert repl.node_type == NodeType.LEAF and repl.params["feature"] == 0
    # binary & root case
    node = __build_tree_from_literals(["*", 2, "-", 0, 1], **hyper_params)
    repl, disc = de_transform(node)
    assert disc.node_type == NodeType.LEAF and disc.params["feature"] == 2
    _assert_tree_completeness(repl, 1, node)
    # binary & non-root case
    node = __build_tree_from_literals(
        ["+", "*", "-", "exp", 0, 1, 2, 3], **hyper_params
    )
    _assert_tree_completeness(node)
    repl, disc = de_transform(node.left)
    if repl is node.left.left:
        assert disc is node.left.right
    else:
        assert disc is node.left.left
    _assert_tree_completeness(repl, 2, node.left)


def test_mcmc_transform():
    ops_name_list, ops_weight_list, ops_priors = get_prior_dict()
    hyper_params = {"sigma_a": 1, "sigma_b": 1}

    node = __build_tree_from_literals(["*", "+", "-", 0, 1, 0, 1], **hyper_params)
    old_left = node.left
    transform(old_left, ops_name_list, ops_weight_list, ops_priors, **hyper_params)
    _assert_tree_completeness(node)
    assert old_left.parent.parent is node
    assert node.left is old_left.parent
    assert old_left.parent.left is old_left


def test_mcmc_reassign_op():
    ops_name_list, ops_weight_list, ops_priors = get_prior_dict()
    hyper_params = {"sigma_a": 1, "sigma_b": 1}
    # repeat multiple times to cover all cases
    for _ in range(5):
        node = __build_tree_from_literals(["*", "exp", "-", 0, 0, 1], **hyper_params)
        reassign_op(node, ops_name_list, ops_weight_list, ops_priors, **hyper_params)
        _assert_tree_completeness(node)
        if node.node_type == NodeType.BINARY:
            assert node.left.op_name == "exp"
            assert node.right.op_name == "-"
        else:
            assert node.left.op_name == "exp"
            assert node.left.left.params["feature"] == 0
        reassign_op(
            node.left, ops_name_list, ops_weight_list, ops_priors, **hyper_params
        )
        _assert_tree_completeness(node)
        assert node.left.left.params["feature"] == 0
        if node.left.node_type == NodeType.BINARY:
            assert node.left.right.node_type != NodeType.EMPTY


def test_mcmc_reassign_feat(**hyper_params):
    node = __build_tree_from_literals(["*", "exp", "-", 2, 2, 3], **hyper_params)
    reassign_feat(node.left.left)
    reassign_feat(node.right.right)
    _assert_tree_completeness(node)
    assert node.left.op_name == "exp"
    assert node.left.left.params["feature"] < 2
    assert node.right.left.params["feature"] == 2
    assert node.right.right.params["feature"] < 2
