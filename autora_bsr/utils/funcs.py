import copy

import numpy as np
from autora_bsr.utils.node import Node, NodeType
from typing import Dict, List, Optional, Callable
from scipy.stats import invgamma
from functools import wraps


def check_empty(func: Callable):
    """
    A decorator that, if applied to `func`, checks whether an argument in `func` is an
    un-initialized node (i.e. node.node_type == NodeType.Empty). If so, an error is raised.
    """
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        for arg in args + list(kwargs.values()):
            if isinstance(arg, Node) and arg.node_type == NodeType.EMPTY:
                raise TypeError("uninitialized node found in {}".format(func.__name__))
        return func(*args, **kwargs)
    return func_wrapper


@check_empty
def get_height(node: Node) -> int:
    """
    Get the height of a tree starting from `node` as root. The height of a leaf is defined as 0.

    Arguments:
        node: the Node that we hope to calculate `height` for
    Returns:
        height: the height of `node`
    """
    if node.node_type == NodeType.LEAF:
        return 0
    elif node.node_type == NodeType.UNARY:
        return 1 + get_height(node.left)
    else:  # binary node
        return 1 + max(get_height(node.left), get_height(node.right))


@check_empty
def update_depth(node: Node, depth: int):
    node.depth = depth
    if node.node_type == NodeType.UNARY:
        update_depth(node.left, depth + 1)
    elif node.node_type == NodeType.BINARY:
        update_depth(node.left, depth + 1)
        update_depth(node.right, depth + 1)


def get_expression(
    node: Node,
    ops_expr: Dict[str, str],
    feature_names: Optional[List[str]] = None,
) -> str:
    if node.node_type == NodeType.LEAF:
        if feature_names:
            return feature_names[node.params["feature"]]
        else:
            return f"x{node.params['feature']}"
    elif node.node_type == NodeType.UNARY:
        # if the expr for an operator is not defined, use placeholder
        # e.g. operator `cosh` -> `cosh(xxx)`
        place_holder = node.op_name + "({})"
        left_expr = get_expression(node.left, ops_expr, feature_names)
        expr_fmt = ops_expr.get(node.op_name, place_holder)
        return expr_fmt.format(left_expr, **node.params)
    elif node.node_type == NodeType.BINARY:
        place_holder = node.op_name + "({})"
        left_expr = get_expression(node.left, ops_expr, feature_names)
        right_expr = get_expression(node.right, ops_expr, feature_names)
        expr_fmt = ops_expr.get(node.op_name, place_holder)
        return expr_fmt.format(left_expr, right_expr, **node.params)
    else:  # empty node
        return "(empty node)"


@check_empty
def get_all_nodes(node: Node) -> List[Node]:
    """
    Get all the nodes below (and including) the given `node` via pre-order traversal

    Return:
        a list with all the nodes below (and including) the given `node`
    """
    nodes = [node]
    if node.node_type == NodeType.UNARY:
        nodes.extend(get_all_nodes(node.left))
    elif node.node_type == NodeType.BINARY:
        nodes.extend(get_all_nodes(node.left))
        nodes.extend(get_all_nodes(node.right))
    return nodes


def calc_tree_ll(
    node: Node,
    ops_priors: Dict[str, Dict],
    n_feature: int = 1,
    **hyper_params: Dict
):
    struct_ll = 0  # log likelihood of tree structure S = (T,M)
    params_ll = 0  # log likelihood of linear params
    depth = node.depth
    beta = hyper_params.get("beta", -1)
    sigma_a, sigma_b = hyper_params.get("sigma_a", 1), hyper_params.get("sigma_b", 1)
    op_weight = ops_priors[node.op_name]["weight"]

    # contribution of hyperparameter sigma_theta
    if not depth:  # root node
        struct_ll += np.log(invgamma.pdf(sigma_a, 1))
        struct_ll += np.log(invgamma.pdf(sigma_b, 1))

    # contribution of splitting the node or becoming leaf node
    if node.node_type == NodeType.LEAF:
        # contribution of choosing terminal
        struct_ll += np.log(
            1 - 1 / np.power((1 + depth), -beta)
        )
        # contribution of feature selection
        struct_ll -= np.log(n_feature)
        return struct_ll, params_ll
    elif node.node_type == NodeType.UNARY:  # unitary operator
        # contribution of child nodes are added since the log likelihood is additive
        # if we assume the parameters are independent.
        struct_ll_left, params_ll_left = calc_tree_ll(node.left, ops_priors, n_feature, **hyper_params)
        struct_ll += struct_ll_left
        params_ll += params_ll_left
        # contribution of parameters of linear nodes
        # TODO: make sure the below parameter ll calculation is extendable
        if node.op_name == "ln":
            params_ll -= np.power((node.a - 1), 2) / (2 * sigma_a)
            params_ll -= np.power(node.b, 2) / (2 * sigma_b)
            params_ll -= 0.5 * np.log(4 * np.pi ** 2 * sigma_a * sigma_b)
    else:  # binary operator
        struct_ll_left, params_ll_left = calc_tree_ll(node.right, ops_priors, n_feature, **hyper_params)
        struct_ll_right, params_ll_right = calc_tree_ll(node.right, ops_priors, n_feature, **hyper_params)
        struct_ll += struct_ll_left + struct_ll_right
        params_ll += params_ll_left + params_ll_right

    # for unary & binary nodes, additionally consider the contribution of splitting
    if not depth:  # root node
        struct_ll += np.log(op_weight)
    else:
        struct_ll += np.log((1 + depth)) * beta + np.log(op_weight)

    return struct_ll, params_ll


def stay(ln_nodes: List[Node], **hyper_params: Dict):
    """
    ACTION 1: Stay represents the action of doing nothing but to update the parameters for `ln`
    operators.

    Arguments:
        ln_nodes: the list of nodes with `ln` operator
        hyper_params: hyperparameters for re-initialization
    """
    for ln_node in ln_nodes:
        ln_node.init_param(**hyper_params)


def grow(
    node: Node,
    ops_name_lst: List[str],
    ops_weight_lst: List[float],
    ops_priors: Dict[str, Dict],
    n_feature: int = 1,
    **hyper_params: Dict
):
    """
    ACTION 2: Grow represents the action of growing a subtree from a given `node`

    Arguments:
        node: the tree node from where the subtree starts to grow
        ops_name_lst: list of operation names
        ops_weight_lst: list of operation prior weights
        ops_priors: the dictionary of operation prior properties
        n_feature: the number of features in input data
        hyper_params: hyperparameters for re-initialization
    """
    depth = node.depth
    p = 1 / np.power((1 + depth), -hyper_params.get("beta", -1))

    if depth > 0 and p < np.random.uniform(0, 1, 1):  # create leaf node
        node.setup(feature=np.random.randint(0, n_feature, 1))
    else:
        ops_name = np.random.choice(ops_name_lst, p=ops_weight_lst)
        ops_prior = ops_priors[ops_name]
        node.setup(ops_name, ops_prior, **hyper_params)

        # recursively set up downstream nodes
        grow(node.left, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)
        if node.node_type == NodeType.BINARY:
            grow(node.right, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)


@check_empty
def prune(node: Node, n_feature: int = 1):
    """
    ACTION 3: Prune a non-terminal node into a terminal node and assign it a feature

    Arguments:
        node: the tree node to be pruned
        n_feature: the number of features in input data
    """
    node.setup(feature=np.random.randint(0, n_feature, 1))


@check_empty
def de_transform(node: Node) -> Node:
    """
    ACTION 4: De-transform deletes the current `node` and replaces it with children
    according to the following rule: if the `node` is unary, simply replace with its
    child; if `node` is binary and root, choose any children that's not leaf; if `node`
    is binary and not root, pick any children.

    Arguments:
        node: the tree node that gets de-transformed
    """
    if node.node_type == NodeType.UNARY:
        return node.left
    r = np.random.random()
    # picked node is root
    if not node.depth:
        if node.left.node_type == NodeType.LEAF:
            return node.right
        elif node.right.node_type == NodeType.LEAF:
            return node.left
        else:
            return node.left if r < 0.5 else node.right
    elif r < 0.5:
        return node.left
    else:
        return node.right


@check_empty
def transform(
    node: Node,
    ops_name_lst: List[str],
    ops_weight_lst: List[float],
    ops_priors: Dict[str, Dict],
    n_feature: int = 1,
    **hyper_params: Dict
):
    assert node.parent is not None
    parent = node.parent
    is_left = node is parent.left

    insert_node = Node(depth=node.depth, parent=parent)
    insert_op = np.random.choice(ops_name_lst, 1, ops_weight_lst)[0]
    insert_node.setup(insert_op, ops_priors[insert_op], **hyper_params)

    if is_left:
        parent.left = insert_node
    else:
        parent.right = insert_node

    # set the left child as `node` and grow the right child if needed (binary case)
    insert_node.left = node
    if insert_node.node_type == NodeType.BINARY:
        grow(insert_node.right, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)

    # make sure the depth property is updated correctly
    update_depth(node, node.depth + 1)


@check_empty
def prop(node: Node):
    """
    Propose a new tree from an existing tree with root `node`

    Return:
    """
    # PART 1: collect necessary information
    new_node = copy.deepcopy(node)
    term_nodes, nterm_nodes, ln_nodes = [], [], []
    for n in get_all_nodes(new_node):
        if n.node_type == NodeType.LEAF:
            term_nodes.append(n)
        else:
            nterm_nodes.append(n)
        if n.op_name == "ln":
            ln_nodes.append(n)

