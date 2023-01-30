from autora_bsr.utils.node import Node
from autora_bsr.utils.funcs import grow, get_expression
from autora_bsr.utils.prior import get_prior_dict
from autora_bsr.utils.misc import get_ops_expr


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
    print(get_expression(root, ops_expr))
