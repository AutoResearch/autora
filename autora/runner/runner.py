from typing import Callable

import numpy as np


def runner(condition, set_condition: Callable, get_dependent: Callable, **kwargs):
    """
    Generic experiment runner. To construct a specific, you need to define a set_condition and
    get_dependent function.
    Args:
        condition: the condition that the runner should run
        set_condition: a function to setup the experiment with the condition
        get_dependent: a function to get the dependent variable
        **kwargs:

    Returns:
        The dependent variable

    """
    set_condition(condition, **kwargs)
    return np.array(get_dependent(**kwargs)).reshape(condition.shape)
