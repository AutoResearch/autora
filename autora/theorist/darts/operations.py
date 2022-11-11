import typing
from collections import namedtuple

import torch
import torch.nn as nn

Genotype = namedtuple("Genotype", "normal normal_concat")


def isiterable(p_object: typing.Any) -> bool:
    """
    Checks if an object is iterable.

    Arguments:
        p_object: object to be checked
    """
    try:
        iter(p_object)
    except TypeError:
        return False
    return True


def get_operation_label(
    op_name: str,
    params_org: typing.List,
    decimals: int = 4,
    input_var: str = "x",
    output_format: typing.Literal["latex", "console"] = "console",
) -> str:
    r"""
    Returns a complete string describing a DARTS operation.

    Arguments:
        op_name: name of the operation
        params_org: original parameters of the operation
        decimals: number of decimals to be used for converting the parameters into string format
        input_var: name of the input variable
        output_format: format of the output string (either "latex" or "console")

    Examples:
        >>> get_operation_label("classifier", [1], decimals=2)
        '1.00 * x'
        >>> import numpy as np
        >>> print(get_operation_label("classifier_concat", np.array([1, 2, 3]),
        ...     decimals=2, output_format="latex"))
        x \circ \left(1.00\right) + \left(2.00\right) + \left(3.00\right)
        >>> get_operation_label("classifier_concat", np.array([1, 2, 3]),
        ...     decimals=2, output_format="console")
        'x .* (1.00) .+ (2.00) .+ (3.00)'
        >>> get_operation_label("linear_exp", [1,2], decimals=2)
        'exp(1.00 * x + 2.00)'
        >>> get_operation_label("none", [])
        ''
        >>> get_operation_label("reciprocal", [1], decimals=0)
        '1 / x'
        >>> get_operation_label("linear_reciprocal", [1, 2], decimals=0)
        '1 / (1 * x + 2)'
        >>> get_operation_label("linear_relu", [1], decimals=0)
        'ReLU(1 * x)'
        >>> print(get_operation_label("linear_relu", [1], decimals=0, output_format="latex"))
        \operatorname{ReLU}\left(1x\right)
        >>> get_operation_label("linear", [1, 2], decimals=0)
        '1 * x + 2'
        >>> get_operation_label("linear", [1, 2], decimals=0, output_format="latex")
        '1 x + 2'
        >>> get_operation_label("linrelu", [1], decimals=0)  # Mistyped operation name
        Traceback (most recent call last):
        ...
        NotImplementedError: operation 'linrelu' is not defined for output_format 'console'
    """
    if output_format != "latex" and output_format != "console":
        raise ValueError("output_format must be either 'latex' or 'console'")

    params = params_org.copy()

    format_string = "{:." + "{:.0f}".format(decimals) + "f}"

    classifier_str = ""
    if op_name == "classifier":
        value = params[0]
        classifier_str = f"{format_string.format(value)} * {input_var}"
        return classifier_str

    if op_name == "classifier_concat":
        if output_format == "latex":
            classifier_str = input_var + " \\circ \\left("
        else:
            classifier_str = input_var + " .* ("
        for param_idx, param in enumerate(params):

            if param_idx > 0:
                if output_format == "latex":
                    classifier_str += " + \\left("
                else:
                    classifier_str += " .+ ("

            if isiterable(param.tolist()):

                param_formatted = list()
                for value in param.tolist():
                    param_formatted.append(format_string.format(value))

                for value_idx, value in enumerate(param_formatted):
                    if value_idx < len(param) - 1:
                        classifier_str += value + " + "
                    else:
                        if output_format == "latex":
                            classifier_str += value + "\\right)"
                        else:
                            classifier_str += value + ")"

            else:
                value = format_string.format(param)

                if output_format == "latex":
                    classifier_str += value + "\\right)"
                else:
                    classifier_str += value + ")"

        return classifier_str

    num_params = len(params)

    c = [str(format_string.format(p)) for p in params_org]
    c.extend(["", "", ""])

    if num_params == 1:  # without bias
        if output_format == "console":
            labels = {
                "none": "",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} * {input_var}",
                "linear": f"{c[0]} * {input_var}",
                "relu": f"ReLU({input_var})",
                "linear_relu": f"ReLU({c[0]} * {input_var})",
                "logistic": f"logistic({input_var})",
                "linear_logistic": f"logistic({c[0]} * {input_var})",
                "exp": f"exp({input_var})",
                "linear_exp": f"exp({c[0]} * {input_var})",
                "reciprocal": f"1 / {input_var}",
                "linear_reciprocal": f"1 / ({c[0]} * {input_var})",
                "ln": f"ln({input_var})",
                "linear_ln": f"ln({c[0]} * {input_var})",
                "cos": f"cos({input_var})",
                "linear_cos": f"cos({c[0]} * {input_var})",
                "sin": f"sin({input_var})",
                "linear_sin": f"sin({c[0]} * {input_var})",
                "tanh": f"tanh({input_var})",
                "linear_tanh": f"tanh({c[0]} * {input_var})",
                "classifier": classifier_str,
            }
        elif output_format == "latex":
            labels = {
                "none": "",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} {input_var}",
                "linear": c[0] + "" + input_var,
                "relu": f"\\operatorname{{ReLU}}\\left({input_var}\\right)",
                "linear_relu": f"\\operatorname{{ReLU}}\\left({c[0]}{input_var}\\right)",
                "logistic": f"\\sigma\\left({input_var}\\right)",
                "linear_logistic": f"\\sigma\\left({c[0]} {input_var} \\right)",
                "exp": f"+ e^{input_var}",
                "linear_exp": f"e^{{{c[0]} {input_var} }}",
                "reciprocal": f"\\frac{{1}}{{{input_var}}}",
                "linear_reciprocal": f"\\frac{{1}}{{{c[0]} {input_var} }}",
                "ln": f"\\ln\\left({input_var}\\right)",
                "linear_ln": f"\\ln\\left({c[0]} {input_var} \\right)",
                "cos": f"\\cos\\left({input_var}\\right)",
                "linear_cos": f"\\cos\\left({c[0]} {input_var} \\right)",
                "sin": f"\\sin\\left({input_var}\\right)",
                "linear_sin": f"\\sin\\left({c[0]} {input_var} \\right)",
                "tanh": f"\\tanh\\left({input_var}\\right)",
                "linear_tanh": f"\\tanh\\left({c[0]} {input_var} \\right)",
                "classifier": classifier_str,
            }
    else:  # with bias
        if output_format == "console":
            labels = {
                "none": "",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} * {input_var}",
                "linear": f"{c[0]} * {input_var} + {c[1]}",
                "relu": f"ReLU({input_var})",
                "linear_relu": f"ReLU({c[0]} * {input_var} + {c[1]} )",
                "logistic": f"logistic({input_var})",
                "linear_logistic": f"logistic({c[0]} * {input_var} + {c[1]})",
                "exp": f"exp({input_var})",
                "linear_exp": f"exp({c[0]} * {input_var} + {c[1]})",
                "reciprocal": f"1 / {input_var}",
                "linear_reciprocal": f"1 / ({c[0]} * {input_var} + {c[1]})",
                "ln": f"ln({input_var})",
                "linear_ln": f"ln({c[0]} * {input_var} + {c[1]})",
                "cos": f"cos({input_var})",
                "linear_cos": f"cos({c[0]} * {input_var} + {c[1]})",
                "sin": f"sin({input_var})",
                "linear_sin": f"sin({c[0]} * {input_var} + {c[1]})",
                "tanh": f"tanh({input_var})",
                "linear_tanh": f"tanh({c[0]} * {input_var} + {c[1]})",
                "classifier": classifier_str,
            }
        elif output_format == "latex":
            labels = {
                "none": "",
                "add": f"+ {input_var}",
                "subtract": f"- {input_var}",
                "mult": f"{c[0]} * {input_var}",
                "linear": f"{c[0]} {input_var} + {c[1]}",
                "relu": f"\\operatorname{{ReLU}}\\left( {input_var}\\right)",
                "linear_relu": f"\\operatorname{{ReLU}}\\left({c[0]}{input_var} + {c[1]} \\right)",
                "logistic": f"\\sigma\\left( {input_var} \\right)",
                "linear_logistic": f"\\sigma\\left( {c[0]} {input_var} + {c[1]} \\right)",
                "exp": f"e^{input_var}",
                "linear_exp": f"e^{{ {c[0]} {input_var} + {c[1]} }}",
                "reciprocal": f"\\frac{{1}}{{{input_var}}}",
                "linear_reciprocal": f"\\frac{{1}} {{ {c[0]}{input_var} + {c[1]} }}",
                "ln": f"\\ln\\left({input_var}\\right)",
                "linear_ln": f"\\ln\\left({c[0]} {input_var} + {c[1]} \\right)",
                "cos": f"\\cos\\left({input_var}\\right)",
                "linear_cos": f"\\cos\\left({c[0]} {input_var} + {c[1]} \\right)",
                "sin": f"\\sin\\left({input_var}\\right)",
                "linear_sin": f"\\sin\\left({c[0]} {input_var} + {c[1]} \\right)",
                "tanh": f"\\tanh\\left({input_var}\\right)",
                "linear_tanh": f"\\tanh\\left({c[0]} {input_var} + {c[1]} \\right)",
                "classifier": classifier_str,
            }

    if op_name not in labels:
        raise NotImplementedError(
            f"operation '{op_name}' is not defined for output_format '{output_format}'"
        )

    return labels[op_name]


class Identity(nn.Module):
    """
    A pytorch module implementing the identity function.

    $$
    x = x
    $$
    """

    def __init__(self):
        """
        Initializes the identify function.
        """
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the identity function.

        Arguments:
            x: input tensor
        """
        return x


class NegIdentity(nn.Module):
    """
    A pytorch module implementing the inverse of an identity function.

    $$
    x = -x
    $$
    """

    def __init__(self):
        """
        Initializes the inverse of an identity function.
        """
        super(NegIdentity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the inverse of an identity function.

        Arguments:
            x: input tensor
        """
        return -x


class Exponential(nn.Module):
    """
    A pytorch module implementing the exponential function.

    $$
    x = e^x
    $$
    """

    def __init__(self):
        """
        Initializes the exponential function.
        """
        super(Exponential, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the exponential function.

        Arguments:
            x: input tensor
        """
        return torch.exp(x)


class Cosine(nn.Module):
    r"""
    A pytorch module implementing the cosine function.

    $$
    x = \cos(x)
    $$
    """

    def __init__(self):
        """
        Initializes the cosine function.
        """
        super(Cosine, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cosine function.

        Arguments:
            x: input tensor
        """
        return torch.cos(x)


class Sine(nn.Module):
    r"""
    A pytorch module implementing the sine function.

    $$
    x = \sin(x)
    $$
    """

    def __init__(self):
        """
        Initializes the sine function.
        """
        super(Sine, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sine function.

        Arguments:
            x: input tensor
        """
        return torch.sin(x)


class Tangens_Hyperbolicus(nn.Module):
    r"""
    A pytorch module implementing the tangens hyperbolicus function.

    $$
    x = \tanh(x)
    $$
    """

    def __init__(self):
        """
        Initializes the tangens hyperbolicus function.
        """
        super(Tangens_Hyperbolicus, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the tangens hyperbolicus function.

        Arguments:
            x: input tensor
        """
        return torch.tanh(x)


class NatLogarithm(nn.Module):
    r"""
    A pytorch module implementing the natural logarithm function.

    $$
    x = \ln(x)
    $$

    """

    def __init__(self):
        """
        Initializes the natural logarithm function.
        """
        super(NatLogarithm, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the natural logarithm function.

        Arguments:
            x: input tensor
        """
        # make sure x is in domain of natural logarithm
        mask = x.clone()
        mask[(x <= 0.0).detach()] = 0
        mask[(x > 0.0).detach()] = 1

        epsilon = 1e-10
        result = torch.log(nn.functional.relu(x) + epsilon) * mask

        return result


class MultInverse(nn.Module):
    r"""
    A pytorch module implementing the multiplicative inverse.

    $$
    x = \frac{1}{x}
    $$
    """

    def __init__(self):
        """
        Initializes the multiplicative inverse.
        """
        super(MultInverse, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multiplicative inverse.

        Arguments:
            x: input tensor
        """
        return torch.pow(x, -1)


class Zero(nn.Module):
    """
    A pytorch module implementing the zero operation (i.e., a null operation). A zero operation
    presumes that there is no relationship between the input and output.

    $$
    x = 0
    $$
    """

    def __init__(self, stride):
        """
        Initializes the zero operation.
        """
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the zero operation.

        Arguments:
            x: input tensor
        """
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class Softplus(nn.Module):
    r"""
    A pytorch module implementing the softplus function:

    $$
    \operatorname{Softplus}(x) = \frac{1}{β} \operatorname{log} \left( 1 + e^{β x} \right)
    $$
    """

    # This docstring is a raw-string (it starts `r"""` rather than `"""`)
    # so backslashes need not be escaped

    def __init__(self):
        """
        Initializes the softplus function.
        """
        super(Softplus, self).__init__()
        # self.beta = nn.Linear(1, 1, bias=False)
        self.beta = nn.Parameter(torch.ones(1))
        # elf.softplus = nn.Softplus(beta=self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the softplus function.

        Arguments:
            x: input tensor
        """
        y = torch.log(1 + torch.exp(self.beta * x)) / self.beta
        # y = self.softplus(x)
        return y


class Softminus(nn.Module):
    """
    A pytorch module implementing the softminus function:

    $$
    \\operatorname{Softminus}(x) = x - \\operatorname{log} \\left( 1 + e^{β x} \\right)
    $$
    """

    # This docstring is a normal string, so backslashes need to be escaped

    def __init__(self):
        """
        Initializes the softminus function.
        """
        super(Softminus, self).__init__()
        # self.beta = nn.Linear(1, 1, bias=False)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the softminus function.

        Arguments:
            x: input tensor
        """
        y = x - torch.log(1 + torch.exp(self.beta * x)) / self.beta
        return y


# defines all the operations. affine is turned off for cuda (optimization prposes)
OPS = {
    "none": Zero(1),
    "add": nn.Sequential(Identity()),
    "subtract": nn.Sequential(NegIdentity()),
    "mult": nn.Sequential(
        nn.Linear(1, 1, bias=False),
    ),
    "linear": nn.Sequential(nn.Linear(1, 1, bias=True)),
    "relu": nn.Sequential(
        nn.ReLU(inplace=False),
    ),
    "linear_relu": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        nn.ReLU(inplace=False),
    ),
    "logistic": nn.Sequential(
        nn.Sigmoid(),
    ),
    "linear_logistic": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        nn.Sigmoid(),
    ),
    "exp": nn.Sequential(
        Exponential(),
    ),
    "linear_exp": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        Exponential(),
    ),
    "cos": nn.Sequential(
        Cosine(),
    ),
    "linear_cos": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        Cosine(),
    ),
    "sin": nn.Sequential(
        Sine(),
    ),
    "linear_sin": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        Sine(),
    ),
    "tanh": nn.Sequential(
        Tangens_Hyperbolicus(),
    ),
    "linear_tanh": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        Tangens_Hyperbolicus(),
    ),
    "reciprocal": nn.Sequential(
        MultInverse(),
    ),
    "linear_reciprocal": nn.Sequential(
        nn.Linear(1, 1, bias=False),
        MultInverse(),
    ),
    "ln": nn.Sequential(
        NatLogarithm(),
    ),
    "linear_ln": nn.Sequential(
        nn.Linear(1, 1, bias=False),
        NatLogarithm(),
    ),
    "softplus": nn.Sequential(
        Softplus(),
    ),
    "linear_softplus": nn.Sequential(
        nn.Linear(1, 1, bias=False),
        Softplus(),
    ),
    "softminus": nn.Sequential(
        Softminus(),
    ),
    "linear_softminus": nn.Sequential(
        nn.Linear(1, 1, bias=False),
        Softminus(),
    ),
}

# this is the list of primitives actually used,
# and it should be a set of names contained in the OPS dictionary
PRIMITIVES = (
    "none",
    "add",
    "subtract",
    "linear",
    "linear_logistic",
    "mult",
    "linear_relu",
)

# make sure that every primitive is in the OPS dictionary
for name in PRIMITIVES:
    assert name in OPS
