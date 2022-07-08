import torch
import torch.nn as nn

# defines all the operations. affine is turned off for cuda (optimization prposes)
OPS = {
    "none": lambda affine: Zero(1),
    "linear": lambda affine: nn.Sequential(nn.Linear(1, 1, bias=True)),
    "relu": lambda affine: nn.Sequential(
        nn.ReLU(inplace=False),
    ),
    "lin_relu": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=True),
        nn.ReLU(inplace=False),
    ),
    "sigmoid": lambda affine: nn.Sequential(
        nn.Sigmoid(),
    ),
    "lin_sigmoid": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=True),
        nn.Sigmoid(),
    ),
    "add": lambda affine: nn.Sequential(Identity()),
    "subtract": lambda affine: nn.Sequential(NegIdentity()),
    "mult": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=False),
    ),
    "exp": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=True),
        Exponential(),
    ),
    "1/x": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=False),
        MultInverse(),
    ),
    "ln": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=False),
        NatLogarithm(),
    ),
    "softplus": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=False),
        Softplus(),
    ),
    "softminus": lambda affine: nn.Sequential(
        nn.Linear(1, 1, bias=False),
        Softminus(),
    ),
}


def isiterable(p_object):
    try:
        iter(p_object)
    except TypeError:
        return False
    return True


def get_operation_label(op_name, params_org, decimals=4):
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
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NegIdentity(nn.Module):
    def __init__(self):
        super(NegIdentity, self).__init__()

    def forward(self, x):
        return -x


class Exponential(nn.Module):
    def __init__(self):
        super(Exponential, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class NatLogarithm(nn.Module):
    def __init__(self):
        super(NatLogarithm, self).__init__()

    def forward(self, x):
        # make sure x is in domain of natural logarithm
        mask = x.clone()
        mask[(x <= 0.0).detach()] = 0
        mask[(x > 0.0).detach()] = 1

        epsilon = 1e-10
        result = torch.log(nn.functional.relu(x) + epsilon) * mask

        return result


class MultInverse(nn.Module):
    def __init__(self):
        super(MultInverse, self).__init__()

    def forward(self, x):
        return torch.pow(x, -1)


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


# Softplus(x) = 1/β∗log(1+exp(β∗x))
class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()
        # self.beta = nn.Linear(1, 1, bias=False)
        self.beta = nn.Parameter(torch.ones(1))
        # elf.softplus = nn.Softplus(beta=self.beta)

    def forward(self, x):
        y = torch.log(1 + torch.exp(self.beta * x)) / self.beta
        # y = self.softplus(x)
        return y


class Softminus(nn.Module):
    def __init__(self):
        super(Softminus, self).__init__()
        # self.beta = nn.Linear(1, 1, bias=False)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        y = x - torch.log(1 + torch.exp(self.beta * x)) / self.beta
        return y
