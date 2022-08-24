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
    op_name: str, params_org: typing.List, decimals: int = 4
) -> str:
    """
    Returns a complete string describing a DARTS operation.

    Arguments:
        op_name: name of the operation
        params_org: original parameters of the operation
        decimals: number of decimals to be used for converting the parameters into string format
    """
    params = params_org.copy()

    format_string = "{:." + "{:.0f}".format(decimals) + "f}"

    classifier_str = ""
    if op_name == "classifier":
        value = params[0]
        classifier_str = format_string.format(value) + " * x"

        return classifier_str

    if op_name == "classifier_concat":
        classifier_str = "x.*("
        for param_idx, param in enumerate(params):

            if param_idx > 0:
                classifier_str = classifier_str + " .+("

            if isiterable(param.tolist()):
                for value_idx, value in enumerate(param.tolist()):
                    if value_idx < len(param) - 1:
                        classifier_str = (
                            classifier_str + format_string.format(value) + " + "
                        )
                    else:
                        classifier_str = (
                            classifier_str + format_string.format(value) + ")"
                        )

            else:
                classifier_str = classifier_str + format_string.format(param) + ")"

        return classifier_str

    num_params = len(params)
    params.extend([0, 0, 0])

    if num_params == 1:  # without bias
        labels = {
            "none": "",
            "linear": str(format_string.format(params[0])) + " * x",
            "relu": "ReLU(x)",
            "lin_relu": "ReLU(" + str(format_string.format(params[0])) + " * x)",
            # 'sigmoid': '1/(1+e^(-x))',
            "sigmoid": "logistic(x)",
            # 'lin_sigmoid': '1/(1+e^(-' + str(format_string.format(params[0])) + ' * x))',
            "lin_sigmoid": "logistic(" + str(format_string.format(params[0])) + " * x)",
            "add": "+ x",
            "subtract": "- x",
            "mult": str(format_string.format(params[0])) + " * x",
            # 'exp': 'e^(' + str(format_string.format(params[0])) + ' * x)',
            "exp": "exp(" + str(format_string.format(params[0])) + " * x)",
            "1/x": "1 / (" + str(format_string.format(params[0])) + " * x)",
            "ln": "ln(" + str(format_string.format(params[0])) + " * x)",
            "classifier": classifier_str,
        }
    else:  # with bias
        labels = {
            "none": "",
            "linear": str(format_string.format(params[0]))
            + " * x + "
            + str(format_string.format(params[1])),
            "relu": "ReLU(x)",
            "lin_relu": "ReLU("
            + str(format_string.format(params[0]))
            + " * x + "
            + str(format_string.format(params[1]))
            + ")",
            # 'sigmoid': '1/(1+e^(-x))',
            "sigmoid": "logistic(x)",
            "lin_sigmoid": "logistic("
            + str(format_string.format(params[0]))
            + " * x + "
            + str(format_string.format(params[1]))
            + ")",
            "add": "+ x",
            "subtract": "- x",
            "mult": str(format_string.format(params[0])) + " * x",
            "exp": "exp("
            + str(format_string.format(params[0]))
            + " * x + "
            + str(format_string.format(params[1]))
            + ")",
            "1/x": "1 / ("
            + str(format_string.format(params[0]))
            + " * x + "
            + str(format_string.format(params[1]))
            + ")",
            "ln": "ln("
            + str(format_string.format(params[0]))
            + " * x + "
            + str(format_string.format(params[1]))
            + ")",
            "classifier": classifier_str,
        }

    return labels.get(op_name, "")


class Identity(nn.Module):
    """
    A pytorch module implementing the identity function.
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


class NatLogarithm(nn.Module):
    """
    A pytorch module implementing the natural logarithm function.
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
    """
    A pytorch module implementing the multiplicative inverse.
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
    A pytorch module implementing the softminus function: Softminus(x) = x- log(1+exp(β∗x)).

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
    "linear": nn.Sequential(nn.Linear(1, 1, bias=True)),
    "relu": nn.Sequential(
        nn.ReLU(inplace=False),
    ),
    "lin_relu": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        nn.ReLU(inplace=False),
    ),
    "sigmoid": nn.Sequential(
        nn.Sigmoid(),
    ),
    "lin_sigmoid": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        nn.Sigmoid(),
    ),
    "add": nn.Sequential(Identity()),
    "subtract": nn.Sequential(NegIdentity()),
    "mult": nn.Sequential(
        nn.Linear(1, 1, bias=False),
    ),
    "exp": nn.Sequential(
        nn.Linear(1, 1, bias=True),
        Exponential(),
    ),
    "1/x": nn.Sequential(
        nn.Linear(1, 1, bias=False),
        MultInverse(),
    ),
    "ln": nn.Sequential(
        nn.Linear(1, 1, bias=False),
        NatLogarithm(),
    ),
    "softplus": nn.Sequential(
        nn.Linear(1, 1, bias=False),
        Softplus(),
    ),
    "softminus": nn.Sequential(
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
    "lin_sigmoid",
    "mult",
    "lin_relu",
)

# make sure that every primitive is in the OPS dictionary
for name in PRIMITIVES:
    assert name in OPS
