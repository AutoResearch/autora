from autora.theorist.bsr.funcs import grow
from autora.theorist.bsr.misc import get_ops_expr
from autora.theorist.bsr.node import Node
from autora.theorist.bsr.prior import get_prior_dict


def test_grow_and_print_node():
    root = Node(depth=0)
    ops_name_lst, ops_weight_lst, prior_dict = get_prior_dict()
    hyper_params = {
        "beta": -1,
        "sigma_a": 1,
        "sigma_b": 1,
    }
    grow(root, ops_name_lst, ops_weight_lst, prior_dict, **hyper_params)
    ops_expr = get_ops_expr()
    print(root.get_expression(ops_expr))
