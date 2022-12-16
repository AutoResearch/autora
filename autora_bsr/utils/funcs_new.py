import numpy as np
from autora_bsr.utils.node import Node, NodeType
from typing import Dict, List, Optional


def grow(
    node: Node,
    ops_name_lst: List[str],
    ops_weight_lst: List[float],
    ops_priors: Dict[str, Dict],
    n_feature: int = 1,
    **hyper_params: Dict
):
    depth = node.depth
    params = {}
    p = 1 / np.power((1 + depth), -hyper_params.get("beta", 1))

    if depth > 0 and p < np.random.uniform(0, 1, 1):  # create leaf node
        params["feature"] = np.random.randint(0, n_feature, 1)
        node.setup(**params)
    else:
        ops_name = np.random.choice(ops_name_lst, p=ops_weight_lst)
        ops_prior = ops_priors[ops_name]
        ops_init = ops_prior["init"] or {}
        # init is a function randomized by some hyper-params
        if callable(ops_init):
            params.update(ops_init(**hyper_params))
        else:  # init is deterministic dict
            params.update(ops_init)
        node.setup(ops_name, ops_prior["fn"], ops_prior["arity"], **params)

        # recursively set up downstream nodes
        grow(node.left, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)
        if node.node_type == NodeType.BINARY:
            grow(node.right, ops_name_lst, ops_weight_lst, ops_priors, n_feature, **hyper_params)


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
    else: # empty node
        return "(empty node)"


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
    elif node.node_type == NodeType.EMPTY:
        raise NotImplementedError("cannot get all nodes from uninitialized root")
    return nodes


