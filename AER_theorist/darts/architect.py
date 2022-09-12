import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from AER_theorist.darts.model_search import Network, DARTS_Type

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    # set parameters for architecture learning
    self.network_momentum = args.momentum
    self.network_weight_decay = args.arch_weight_decay
    if hasattr(args, 'arch_weight_decay_df'):
      self.network_weight_decay_df = args.arch_weight_decay_df
    else:
      self.network_weight_decay_df = 0

    if hasattr(args, 'arch_weight_decay_base'):
      self.arch_weight_decay_base = args.arch_weight_decay_base * model._steps
    else:
      self.arch_weight_decay_base = 0

    if hasattr(args, 'fair_darts_loss_weight'):
      self.fair_darts_loss_weight = args.fair_darts_loss_weight
    else:
      self.fair_darts_loss_weight = 1

    self.model = model
    self.lr = args.arch_learning_rate
    # architecture is optimized using Adam
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    # initialize weight decay matrix
    self._init_decay_weights()

  def _init_decay_weights(self):

    n_params = list()
    for operation in self.model.cells._ops[0]._ops:
      if Network.isiterable(operation):
        n_params_total = 1    # any non-zero operation is counted as an additional parameter
        for subop in operation:
          for parameter in subop.parameters():
            if (parameter.requires_grad == True):
              n_params_total += parameter.data.numel()
      else:
        n_params_total = 0   # no operation gets zero parameters
      n_params.append(n_params_total)

    self.decay_weights = Variable(torch.zeros(self.model.arch_parameters()[0].data.shape))
    for idx, param in enumerate(n_params):
      if param > 0:
        self.decay_weights[:, idx] = param * self.network_weight_decay_df + self.arch_weight_decay_base
      else:
        self.decay_weights[:, idx] = param
    self.decay_weights = self.decay_weights
    self.decay_weights = self.decay_weights.data

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_valid, target_valid, network_optimizer, unrolled, input_train=None, target_train=None, eta=1):
    # input_train, target_train only needed for approximation (unrolled=True) of architecture gradient
    # when performing a single weigh update
    #
    # initialize gradients to be zero
    self.optimizer.zero_grad()
    # use different backward step depending on whether to use 2nd order approximation for gradient update
    if unrolled: # probably using eta of parameter update here
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    # move Adam one step
    self.optimizer.step()

  # backward step (using first order approximation?)
  def _backward_step(self, input_valid, target_valid):
    if self.model.DARTS_type == DARTS_Type.ORIGINAL:
      loss = self.model._loss(input_valid, target_valid)
    elif self.model.DARTS_type == DARTS_Type.FAIR:
      loss1 = self.model._loss(input_valid, target_valid)
      loss2 = -F.mse_loss(torch.sigmoid(self.model.alphas_normal), 0.5 * torch.ones(self.model.alphas_normal.shape, requires_grad=False)) # torch.tensor(0.5, requires_grad=False)
      loss = loss1 + self.fair_darts_loss_weight * loss2
    else:
      raise Exception("DARTS Type " + str(self.DARTS_type) + " not implemented")

    loss.backward()

    # weight decay proportional to degrees of freedom
    for p in self.model.arch_parameters():
      p.data.sub_((self.decay_weights * self.lr))  # weight decay


  # backward pass using second order approximation
  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):

    # gets the model
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

    if self.model.DARTS_type == DARTS_Type.ORIGINAL:
      unrolled_loss = unrolled_model._loss(input_valid, target_valid)
    elif self.model.DARTS_type == DARTS_Type.FAIR:
      loss1 = self.model._loss(input_valid, target_valid)
      loss2 = -F.mse_loss(torch.sigmoid(self.model.alphas_normal), torch.tensor(0.5, requires_grad=False))
      unrolled_loss = loss1 + self.ifair_darts_loss_weight * loss2
    else:
      raise Exception("DARTS Type " + str(self.DARTS_type) + " not implemented")

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new #.cuda() # Edit SM 10/26/19: uncommented for cuda

  # second order approximation of architecture gradient (see Eqn. 8 from Liu et al, 2019)
  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    # this implements Eqn. 8 from Liu et al. (2019)
    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

