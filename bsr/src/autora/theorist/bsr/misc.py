from typing import Dict

"""
a file for all miscellaneous functions that are used in BSR.
"""


def normalize_prior_dict(prior_dict: Dict[str, float]):
    """
    Normalize the prior weights for the operators so that the weights sum to
    1 and thus can be directly interpreted/used as probabilities.
    """
    prior_sum = 0.0
    for k in prior_dict:
        prior_sum += prior_dict[k]
    if prior_sum > 0:
        for k in prior_dict:
            prior_dict[k] = prior_dict[k] / prior_sum
    else:
        for k in prior_dict:
            prior_dict[k] = 1 / len(prior_dict)


def get_ops_expr() -> Dict[str, str]:
    """
    Get the literal expression for the operation, the `{}` placeholder represents
    an expression that is recursively evaluated from downstream operations. If an
    operator's expression contains additional parameters (e.g. slope/intercept in
    linear operator), write the parameter like `{param}` - the param will be passed
    in using `expr.format(xxx, **params)` format.

    Return:
        A dictionary that maps operator name to its literal expression.
    """
    ops_expr = {
        "neg": "-({})",
        "sin": "sin({})",
        "pow2": "({})^2",
        "pow3": "({})^3",
        "exp": "exp({})",
        "cos": "cos({})",
        "+": "{}+{}",
        "*": "({})*({})",
        "-": "{}-{}",
        "inv": "1/[{}]",
        "linear": "{a}*({})+{b}",
    }
    return ops_expr
