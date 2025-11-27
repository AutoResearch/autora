from typing import Callable, Dict, Union

import numpy as np
from scipy.stats import norm

from .misc import normalize_prior_dict
from .operation import (
    cos_op,
    exp_op,
    inv_op,
    linear_op,
    make_pow_op,
    minus_op,
    multiply_op,
    neg_op,
    plus_op,
    sin_op,
)


def _get_ops_with_arity():
    """
    Get the operator function and arity (number of operands) of each operator.

    Returns:
        ops_fn_and_arity: a dictionary that maps operator name to a list, where
            the first item is the operator function and the second is the number of
            operands that it takes.
    """
    ops_fn_and_arity = {
        "ln": [linear_op, 1],
        "exp": [exp_op, 1],
        "inv": [inv_op, 1],
        "neg": [neg_op, 1],
        "sin": [sin_op, 1],
        "cos": [cos_op, 1],
        "pow2": [make_pow_op(2), 1],
        "pow3": [make_pow_op(3), 1],
        "+": [plus_op, 2],
        "*": [multiply_op, 2],
        "-": [minus_op, 2],
    }
    return ops_fn_and_arity


def linear_init(**hyper_params) -> Dict:
    """
    Initialization function for the linear operator. Two parameters, slope
    (a) and intercept (b) are initialized.

    Arguments:
        hyper_params: the dictionary for hyperparameters. Specifically, this
            function requires `sigma_a` and `sigma_b` to be present.
    Returns:
        a dictionary with initialized `a` and `b` parameters.
    """
    sigma_a, sigma_b = hyper_params.get("sigma_a", 1), hyper_params.get("sigma_b", 1)
    return {
        "a": norm.rvs(loc=1, scale=np.sqrt(sigma_a)),
        "b": norm.rvs(loc=0, scale=np.sqrt(sigma_b)),
    }


def _get_ops_init() -> Dict[str, Union[Callable, object]]:
    """
    Get the initialization functions for operators that require additional
    parameters.

    Returns:
        ops_init: a dictionary that maps operator name to either a parameter
            dict (in the case that the initialization is hard-coded) or an
            initialization function (when it is randomized). The dictionary
            value will be used in growing the `node` (see `funcs_legacy.py`).
    """
    ops_init = {
        "ln": linear_init,
        "inv": {"cutoff": 1e-10},
        "exp": {"cutoff": 1e-10},
    }
    return ops_init


def _get_prior(prior_name: str, prob: bool = True) -> Dict[str, float]:
    prior_dict = {
        "Uniform": {
            "neg": 1.0,
            "sin": 1.0,
            "pow2": 1.0,
            "pow3": 1.0,
            "exp": 1.0,
            "cos": 1.0,
            "+": 1.0,
            "*": 1.0,
            "-": 1.0,
            "inv": 1.0,
            "ln": 1.0,
        },
        "Guimera2020": {
            "neg": 3.350846072163632,
            "sin": 5.965917796154835,
            "pow2": 3.3017352779079734,
            "pow3": 5.9907496760026175,
            "exp": 4.768665265735502,
            "cos": 5.452564657261127,
            "+": 5.808163661224514,
            "*": 5.002213595420244,
            "-": 1.0,  # set arbitrarily now,
            "inv": 1.0,  # set arbitrarily now,
            "ln": 1.0,  # set arbitrarily now,
        },
    }
    assert prior_dict[prior_name] is not None, "prior key not recognized"
    if prob:
        normalize_prior_dict(prior_dict[prior_name])
    return prior_dict[prior_name]


def get_prior_dict(prior_name="Uniform"):
    """
    Get the dictionary of prior information as well as several list of key operator properties

    Argument:
        prior_name: the name of the prior dictionary to use

    Returns:
        ops_name_lst: the list of operator names
        ops_weight_lst: the list of operator weights
        prior_dict: the dictionary of operator prior information
    """
    ops_prior = _get_prior(prior_name)
    ops_init = _get_ops_init()
    ops_fn_and_arity = _get_ops_with_arity()

    ops_name_lst = list(ops_prior.keys())
    ops_weight_lst = list(ops_prior.values())
    prior_dict = {
        k: {
            "init": ops_init.get(k, {}),
            "fn": ops_fn_and_arity[k][0],
            "arity": ops_fn_and_arity[k][1],
            "weight": ops_prior[k],
        }
        for k in ops_prior
    }

    return ops_name_lst, ops_weight_lst, prior_dict


def get_prior_list(prior_name="Uniform"):
    """
    Get a dictionary of key prior properties

    Argument:
        prior_name: the name of the prior dictionary to use

    Returns:
        a dictionary that maps a prior property (e.g. `name`) to the list of such properties
            for each operator.
    """
    ops_prior = _get_prior(prior_name)
    ops_init = _get_ops_init()
    ops_fn_and_arity = _get_ops_with_arity()

    ops_name_lst = list(ops_prior.keys())
    ops_weight_lst = list(ops_prior.values())
    ops_init_lst = [ops_init.get(k, None) for k in ops_name_lst]
    ops_fn_lst = [ops_fn_and_arity[k][0] for k in ops_name_lst]
    ops_arity_lst = [ops_fn_and_arity[k][1] for k in ops_name_lst]
    return {
        "name": ops_name_lst,
        "weight": ops_weight_lst,
        "init": ops_init_lst,
        "fn": ops_fn_lst,
        "arity": ops_arity_lst,
    }
