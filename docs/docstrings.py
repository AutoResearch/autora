""" Example module with docstring content and formatting. """
from typing import NamedTuple, Union

import numpy as np
import scipy.optimize


def first_order_linear(
    x: Union[float, np.ndarray], c: float, m: float
) -> Union[float, np.ndarray]:
    """
    Evaluate a first order linear model of the form y = m x + c.

    Arguments:
        x: input locations on the x-axis
        c: y-intercept of the linear model
        m: gradient of the linear model

    Examples:
        >>> first_order_linear(0. , 1. , 0. )
        1.0
        >>> first_order_linear(np.array([-1. , 0. , 1. ]), c=1.0, m=2.0)
        array([-1.,  1.,  3.])
    """
    y = m * x + c
    return y


class FirstOrderLinearModel(NamedTuple):
    """
    Describes a first order linear model of the form y = m x + c

    Attributes:
        c: y-intercept of the linear model
        m: gradient of the linear model

    Examples:
        >>> model = FirstOrderLinearModel(m=0.5, c=1)
        >>> model(np.array([0., 1., 2.]))
        array([1. , 1.5, 2. ])

    """

    c: float
    m: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the model at locations X.

        Arguments:
            x: locations on the x-axis

        Returns:
            y: values

        """
        y = first_order_linear(x, c=self.c, m=self.m)
        return y


def curve_fitting_function(
    x: np.ndarray,
    y: np.ndarray,
    max_iterations: int = 25,
    starting_m: float = 0,
    starting_c: float = 0,
) -> FirstOrderLinearModel:
    """
    Fits a first order linear model of form y = m x + c using the modified Powell method,
    and ensuring that the fitted model only includes as much precision as is sensible
    from the data.

    Arguments:
        x: input x-values
        y: input y-values
        max_iterations: maximum number of optimization steps to use

    Returns:
        model: Callable object which includes the fitted parameters as attributes.

    Examples:
        >>> x = np.linspace(-1., 1., 100)
        >>> noise = np.random.RandomState(42).normal(loc=0., scale=0.1, size=100)
        >>> y = first_order_linear(x, c=1.234, m=5.678) + noise
        >>> curve_fitting_function(x, y, max_iterations=10)
        FirstOrderLinearModel(c=1.2, m=5.7)
    """

    # The data sometimes have large outliers, making the L2-norm less useful.
    # We use the L1-norm to be more robust to outliers.
    def l1(params):
        l1_norm = np.sum(np.abs(y - first_order_linear(x, c=params[0], m=params[1])))
        return l1_norm

    results = scipy.optimize.minimize(
        fun=l1,
        x0=(starting_c, starting_m),
        method="Powell",
        options=dict(maxiter=max_iterations),
    )

    # The results of the minimizer are floats with a very high precision,
    # potentially much higher than we would be confident reporting.
    # A rule of thumb is that we get one significant figure of precision for each step of 10x
    # in the dataset size. We round the data to that precision.
    significant_figures = np.round(np.log10(x.shape[0])).astype(int)

    model = FirstOrderLinearModel(
        c=_round_significant_figures(results.x[0], significant_figures),
        m=_round_significant_figures(results.x[1], significant_figures),
    )

    return model


def _round_significant_figures(x: float, significant_figures: int):
    """
    Examples:

        # Typical range of values, positive:
        >>> _round_significant_figures(11, 1)
        10.0
        >>> _round_significant_figures(11, 2)
        11.0
        >>> _round_significant_figures(0.11, 1)
        0.1
        >>> _round_significant_figures(0.11, 2)
        0.11
        >>> _round_significant_figures(0.110, 3)
        0.11
        >>> _round_significant_figures(1_234_567, 3)
        1230000.0
        >>> _round_significant_figures(1_234_567, 7)
        1234567.0
        >>> _round_significant_figures(0.1_234_567, 7)
        0.1234567
        >>> _round_significant_figures(1.23456789e12, 1)
        1000000000000.0
        >>> _round_significant_figures(1.23456789e24, 1)
        1e+24
        >>> _round_significant_figures(9_999_999, 1)
        10000000.0
        >>> _round_significant_figures(9_870_000, 3)
        9870000.0

        # Special value: zero, always returns zero
        >>> _round_significant_figures(0.0, 1)
        0.0
        >>> _round_significant_figures(0.0, 100)
        0.0

        # Typical range of values, negative:
        >>> _round_significant_figures(-11, 1)
        -10.0
        >>> _round_significant_figures(-1.234567e6, 1)
        -1000000.0
        >>> _round_significant_figures(-1.234567e6, 5)
        -1234600.0
        >>> _round_significant_figures(-1.234567e-10, 7)
        -1.234567e-10
        >>> _round_significant_figures(-1.234567e-10, 2)
        -1.2e-10
        >>> _round_significant_figures(1e-12, 1)
        1e-12

        # Zero significant figures -> zero.
        >>> _round_significant_figures(0.1_234_567, 0)
        0.0
        >>> _round_significant_figures(11, 0)
        0.0
        >>> _round_significant_figures(1e12, 0)
        0.0
        >>> _round_significant_figures(1e99, 0)
        0.0
        >>> _round_significant_figures(1e-12, 0)
        0.0
        >>> _round_significant_figures(1e-99, 0)
        0.0

        # Numbers outside the domain 1e-23 < x < 1e25 have potential for rounding errors
        >>> _round_significant_figures(1e-99, 1)
        9.999999999999994e-100
        >>> _round_significant_figures(1e-23, 1)
        1.0000000000000001e-23
        >>> _round_significant_figures(1e25, 1)
        9.999999999999999e+24
        >>> _round_significant_figures(1e99, 1)
        1.0000000000000006e+99

        # Errors:
        >>> _round_significant_figures(11, -1)
        Traceback (most recent call last):
        AssertionError: significant_figures -1 is not >= 0



    """
    assert (
        significant_figures >= 0
    ), f"significant_figures {significant_figures} is not >= 0"

    if x == 0.0:
        return x

    exponent = np.floor(np.log10(np.abs(x))).astype(int)

    # We can't round the mantissa directly using numpy.round,
    # so we do it in terms of decimal places.
    # For example: 2 significant figures of 123_456 (= 1.23456e5) are 120_000 (= 1.2e5)
    # This is equivalent to rounding by -4 decimal places:
    # -4 decimal_places = 2 significant_figures - 5 exponent - 1 constant
    decimal_places = significant_figures - exponent - 1

    rounded_x = np.round(x, decimal_places).astype(float)

    return rounded_x


if __name__ == "__main__":
    # Some example calls we can use with a debugger to investigate the operation of the code
    _round_significant_figures(11, 1)
    _round_significant_figures(11, 2)
    _round_significant_figures(0.11, 1)
    _round_significant_figures(0.11, 2)
    _round_significant_figures(0, 2)

    # Example code to see how big or small the numbers can get without losing precision.
    # For visual inspection.
    for e in range(-50, 50):
        x = 10.0**e
        print(f"{x}: {_round_significant_figures(x, 1)}")
