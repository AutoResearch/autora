import torch
import torch.nn as nn
import numpy as np

# defines all the operations. affine is turned off for cuda (optimization prposes)
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'linear' : lambda C, stride, affine: nn.Sequential(
    nn.Linear(1, 1, bias=True)
    ),
  'relu' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    ),
  'lin_relu' : lambda C, stride, affine: nn.Sequential(
    nn.Linear(1, 1, bias=True),
    nn.ReLU(inplace=False),
    ),
  'sigmoid': lambda C, stride, affine: nn.Sequential(
    nn.Sigmoid(),
   ),
  'lin_sigmoid': lambda C, stride, affine: nn.Sequential(
    nn.Linear(1, 1, bias=True),
    nn.Sigmoid(),
   ),
  'add': lambda C, stride, affine: nn.Sequential(
    Identity()
   ),
  'subtract': lambda C, stride, affine: nn.Sequential(
    NegIdentity()
   ),
  'mult': lambda C, stride, affine: nn.Sequential(
    nn.Linear(1, 1, bias=False),
   ),
  # 'exp': lambda C, stride, affine: nn.Sequential(
  #   Exponential(),
  #  ),
  'exp': lambda C, stride, affine: nn.Sequential(
    nn.Linear(1, 1, bias=True),
    Exponential(),
   ),
  '1/x': lambda C, stride, affine: nn.Sequential(
    nn.Linear(1, 1, bias=False),
    MultInverse(),
   ),
  'ln': lambda C, stride, affine: nn.Sequential(
    nn.Linear(1, 1, bias=False),
    NatLogarithm(),
   ),
  # EDIT 11/04/19 SM: adapting to new SimpleNet data
  # 'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  # 'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  # 'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  # 'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  # 'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  # 'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  # 'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  # 'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  # 'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
  #   nn.ReLU(inplace=False),
  #   nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
  #   nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
  #   nn.BatchNorm2d(C, affine=affine)
  #   ),
}


def isiterable(p_object):
  try:
    it = iter(p_object)
  except TypeError:
    return False
  return True

def get_operation_label(op_name, params_org, decimals=4):

  params = params_org.copy()

  format_string = "{:." + "{:.0f}".format(decimals) + "f}"

  classifier_str = ''
  if (op_name == 'classifier'):

    value = params[0]
    classifier_str = format_string.format(value) + ' * x'

    return classifier_str

  if (op_name == 'classifier_concat'):
    classifier_str = 'x.*('
    for param_idx, param in enumerate(params):

      if param_idx > 0:
        classifier_str = classifier_str + ' .+('

      if isiterable(param.tolist()):
        for value_idx, value in enumerate(param.tolist()):
          if value_idx < len(param)-1:
            classifier_str = classifier_str + format_string.format(value) + " + "
          else:
            classifier_str = classifier_str + format_string.format(value) + ")"

      else:
        classifier_str = classifier_str + format_string.format(param) + ")"

    return classifier_str

  num_params = len(params)
  params.extend([0, 0, 0])

  if num_params == 1: # without bias
    labels = {
      'none': '',
      'linear': str(format_string.format(params[0])) + ' * x',
      'relu': 'ReLU(x)',
      'lin_relu': 'ReLU(' + str(format_string.format(params[0])) + ' * x)',
      # 'sigmoid': '1/(1+e^(-x))',
      'sigmoid': 'logistic(x)',
      # 'lin_sigmoid': '1/(1+e^(-' + str(format_string.format(params[0])) + ' * x))',
      'lin_sigmoid': 'logistic(' + str(format_string.format(params[0])) + ' * x)',
      'add': '+ x',
      'subtract': '- x',
      'mult': str(format_string.format(params[0])) + ' * x',
      # 'exp': 'e^(' + str(format_string.format(params[0])) + ' * x)',
      'exp': 'exp(' + str(format_string.format(params[0])) + ' * x)',
      '1/x': '1 / (' + str(format_string.format(params[0])) + ' * x)',
      'ln': 'ln(' + str(format_string.format(params[0])) + ' * x)',
      'classifier': classifier_str
    }
  else: # with bias
    labels = {
      'none': '',
      'linear': str(format_string.format(params[0])) + ' * x + ' + str(format_string.format(params[1])),
      'relu': 'ReLU(x)',
      'lin_relu': 'ReLU(' + str(format_string.format(params[0])) + ' * x + ' + str(format_string.format(params[1])) + ')',
      # 'sigmoid': '1/(1+e^(-x))',
      'sigmoid': 'logistic(x)',
      # 'lin_sigmoid': '1/(1+e^(-(' + str(format_string.format(params[0])) + ' * x + ' + str(format_string.format(params[1])) + ')))',
      'lin_sigmoid': 'logistic(' + str(format_string.format(params[0])) + ' * x + ' + str(
        format_string.format(params[1])) + ')',
      'add': '+ x',
      'subtract': '- x',
      'mult': str(format_string.format(params[0])) + ' * x',
      # 'exp': 'e^(' + str(format_string.format(params[0])) + ' * x + ' + str(format_string.format(params[1])) + ')',
      'exp': 'exp(' + str(format_string.format(params[0])) + ' * x + ' + str(format_string.format(params[1])) + ')',
      '1/x': '1 / (' + str(format_string.format(params[0])) + ' * x + ' + str(format_string.format(params[1])) + ')',
      'ln': 'ln(' + str(format_string.format(params[0])) + ' * x + ' + str(format_string.format(params[1])) + ')',
      'classifier': classifier_str
    }

  return labels.get(op_name, '')

# this module links two cells
# The first and second nodes of cell k are set equal to the outputs
# of cell k − 2 and cell k − 1, respectively, and 1 × 1 convolutions
# are inserted as necessary
class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


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
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


# module is used for reduction operations
class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

