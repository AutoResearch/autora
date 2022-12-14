import numpy as np


def __get_prior(prior_name):
    prior_dict = {
        "GuimeraTest2020": {
            "Nopi_/": 0,
            "Nopi_cosh": 0,
            "Nopi_-": 0,
            "Nopi_sin": 0,
            "Nopi_tan": 0,
            "Nopi_tanh": 0,
            "Nopi_**": 0,
            "Nopi_pow2": 0,
            "Nopi_pow3": 0,
            "Nopi_exp": 0,
            "Nopi_log": 0,
            "Nopi_sqrt": 0,
            "Nopi_cos": 0,
            "Nopi_sinh": 0,
            "Nopi_abs": 0,
            "Nopi_+": 0,
            "Nopi_*": 0,
            "Nopi_fac": 0,
            "Nopi_sig": 0,
            "Nopi_relu": 0,
        },
        "Guimera2020": {
            "Nopi_/": 5.912205942815285,
            "Nopi_cosh": 8.12720511103694,
            "Nopi_-": 3.350846072163632,
            "Nopi_sin": 5.965917796154835,
            "Nopi_tan": 8.127427922862411,
            "Nopi_tanh": 7.799259068142255,
            "Nopi_**": 6.4734429542245495,
            "Nopi_pow2": 3.3017352779079734,
            "Nopi_pow3": 5.9907496760026175,
            "Nopi_exp": 4.768665265735502,
            "Nopi_log": 4.745957377206544,
            "Nopi_sqrt": 4.760686909134266,
            "Nopi_cos": 5.452564657261127,
            "Nopi_sinh": 7.955723540761046,
            "Nopi_abs": 6.333544134938385,
            "Nopi_+": 5.808163661224514,
            "Nopi_*": 5.002213595420244,
            "Nopi_fac": 10.0,
            "Nopi2_*": 1.0,
            "Nopi_sig": 1.0,  # arbitrarily set for now
            "Nopi_relu": 1.0,  # arbitrarily set for now
        },
    }
    assert prior_dict[prior_name] is not None, "prior key not recognized"
    return prior_dict[prior_name]


def __get_ops():
    ops = {
        "sin": 1,
        "cos": 1,
        "tan": 1,
        "exp": 1,
        "log": 1,
        "sinh": 1,
        "cosh": 1,
        "tanh": 1,
        "pow2": 1,
        "pow3": 1,
        "abs": 1,
        "sqrt": 1,
        "fac": 1,
        "-": 1,
        "+": 2,
        "*": 2,
        "/": 2,
        "**": 2,
        "sig": 1,
        "relu": 1,
    }
    return ops


def get_priors(prior="Guimera2020"):
    priors = __get_prior(prior)
    all_ops = __get_ops()
    ops = {k: v for k, v in all_ops.items() if "Nopi_" + k in priors}
    return priors, ops


def relu(x):
    return np.maximum(x, 0)
