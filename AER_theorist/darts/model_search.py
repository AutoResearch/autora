import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import warnings

from AER_theorist.darts.fan_out import Fan_Out
from AER_theorist.darts.operations import *
from AER_theorist.darts.genotypes import PRIMITIVES
from AER_theorist.darts.genotypes import Genotype

from enum import Enum

class DARTS_Type(Enum):
    ORIGINAL = 1        # Liu, Simonyan & Yang (2018). Darts: Differentiable architecture search
    FAIR = 2            # Chu, Zhou, Zhang & Li (2020). Fair darts: Eliminating unfair advantages in differentiable architecture search

# for 2 input nodes, 1 output node and 4 intermediate nodes, there are 14 possible edges (x 8 operations)

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    # loop through all the 8 primitive operations
    for primitive in PRIMITIVES:
      # OPS returns an nn module for a given primitive (defines as a string). each primitive is defined by the channels size and stride
      op = OPS[primitive](C, stride, False)

      # EDIT 11/04/19 SM: adapting to new SimpleNet data
      # if 'pool' in primitive: # add normalization if there is a pooling operation
      #   op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))

      # add the operation
      self._ops.append(op)
      # if primitive == 'relu':
      #   op[0].weight.data.fill_(1.0)
      #   op[0].bias.data.fill_(0.001)

  def forward(self, x, weights):
    # there are 8 weights for all the eight primitives. then it returns the weighted sum of all operations performed on a given input
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, n_input_states, C):
    # The first and second nodes of cell k are set equal to the outputs of
    # cell k − 2 and cell k − 1, respectively, and 1 × 1 convolutions (ReLUConvBN) are inserted as necessary
    super(Cell, self).__init__()

    # set parameters
    self._steps = steps
    self._n_input_states = n_input_states

    # EDIT 11/04/19 SM: adapting to new SimpleNet data (changed from multiplier to steps)
    self._multiplier = steps

    # set operations according to number of modules (empty)
    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(self._n_input_states+i): # 2 refers to the 2 input nodes
        # defines the stride for link between cells
        stride = 1
        # adds a mixed operation (derived from architecture parameters alpha)
        # for 4 intermediate nodes, a total of 14 connections (MixedOps) is added
        op = MixedOp(C, stride)
        # appends cell with mixed operation
        self._ops.append(op)

  def forward(self, input_states, weights):

    # initialize states (activities of each node in the cell)
    states = list()

    # add each input node to the number of states
    for input in input_states:
      states.append(input)

    offset = 0
    # this computes the states from intermediate nodes and adds them to the list of states
    for i in range(self._steps): # compute the state from each node
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    # concatenates the states of the last n (self._multiplier) intermediate nodes to get the output of a cell
    result = torch.cat(states[-self._multiplier:], dim=1)
    return result


class Network(nn.Module):

  # EDIT 11/04/19 SM: adapting to new SimpleNet data (changed steps from 4 to 2)
  def __init__(self, num_classes, criterion, steps=2, n_input_states = 2, architecture_fixed = False, classifier_weight_decay = 0, darts_type=DARTS_Type.ORIGINAL):
    super(Network, self).__init__()
    self.DARTS_type = darts_type
    # set parameters
    self._C = 1                                         # number of channels  EDIT 11/04/19 SM: adapting to new SimpleNet data (set self._C to 1 instead of C)
    self._num_classes = num_classes                     # number of output classes
    self._criterion = criterion                         # optimization criterion (e.g. softmax)
    self._steps = steps                                 # the number of intermediate nodes (4)
    self._n_input_states = n_input_states               # number of input nodes
    self._multiplier = 1                                # the number of internal nodes that get concatenated to the output
    self._dim_output = self._steps
    self._architecture_fixed = architecture_fixed
    self._classifier_weight_decay = classifier_weight_decay

    C_curr = self._C # computes number of output channels for the stem
    # define the stem of the model; REMOVE THIS, EMBED FANOUT HERE
    # EDIT 11/04/19 SM: adapting to new SimpleNet data
    # self.stem = nn.Sequential(
    #   nn.Conv2d(3, C_curr, 3, padding=1, bias=False), # 2d convolution with 3 input channels (image colors) and 3*16 output channels (16 feature maps per channel)
    #   nn.BatchNorm2d(C_curr) #  normalizes 4D input (a mini-batch of 2D inputs with additional channel dimension)
    # )
    self.stem = nn.Sequential(
      Fan_Out(self._n_input_states)
    )

    # EDIT 11/04/19 SM: adapting to new SimpleNet data
    # C_prev, C_curr = C_curr, C # compute number of channels for the next layer

    self.cells = nn.ModuleList() # get list of all current modules (should be empty)

    # generate a cell that undergoes architecture search
    # EDIT 11/04/19 SM: adapting to new SimpleNet data (replaced self._multiplier with num_input states as second argument)
    self.cells = Cell(steps, self._n_input_states, C_curr)

    # compute number of input and output channels for the next layer
    # EDIT 11/04/19 SM: adapting to new SimpleNet data
    # C_prev = C_curr

    # the pooling stencil size (aka kernel size) is determined to be (input_size+target_size-1) // target_size, i.e. rounded up
    # EDIT 11/04/19 SM: adapting to new SimpleNet data
    # self.global_pooling = nn.AdaptiveAvgPool2d(1) # replace this with a simple all to all connection (or something more constrained)

    # last layer is a linear classifier (e.g. with 10 CIFAR classes)
    # EDIT 11/04/19 SM: adapting to new SimpleNet data (change input of nn layer from C_prev to self._dim_output)
    self.classifier = nn.Linear(self._dim_output, num_classes) # make this the number of input states


    # initializes weights of the architecture
    self._initialize_alphas()

  # function for copying the network
  def new(self):
    model_new = Network(self._C, self._num_classes, self._criterion, steps=self._steps)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  # computes forward pass for full network
  def forward(self, input):

    # compute stem first
    # EDIT 11/04/19 SM: adapting to new SimpleNet data
    # input_states = list()
    # for i in range(self._n_input_states): # for now, it applies the same stem operation to the input n times and retrieves states for separate nodes for the cell
    #   input_states.append(self.stem(input))
    input_states = self.stem(input)

    # get architecture weights
    if self._architecture_fixed:
      weights = self.alphas_normal
    else:
      if self.DARTS_type==DARTS_Type.ORIGINAL:
        weights = F.softmax(self.alphas_normal, dim=-1)
      elif self.DARTS_type==DARTS_Type.FAIR:
        weights = torch.sigmoid(self.alphas_normal)
      else:
        raise Exception("DARTS Type " + str(self.DARTS_type) + " not implemented")

      # then apply cell with weights
    # input_states = [s0, s1]
    cell_output = self.cells(input_states, weights)

    # pool last layer
    # EDIT 11/04/19 SM: adapting to new SimpleNet data
    # out = self.global_pooling(cell_output)

    # compute logits
    logits = self.classifier(cell_output.view(cell_output.size(0),-1)) # just gets output to have only 2 dimensions (batch_size x num units in output layer)

    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) # returns cross entropy

  def apply_weight_decay_to_classifier(self, lr):
    # weight decay proportional to degrees of freedom
    for p in self.classifier.parameters():
      p.data.sub_(self._classifier_weight_decay * lr * torch.sign(p.data) * (torch.abs(p.data)))  # weight decay

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(self._n_input_states+i)) # the number of possible connections between nodes
    # number of available primitive operations (8 different types for a conv net)
    num_ops = len(PRIMITIVES)

    # generate 14 (umber of available edges) by 8 (operations) weight matrix for normal alphas of the architecture
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    # those are all the parameters of the architecture
    self._arch_parameters = [self.alphas_normal]

  # provide back the architecture as a parameter
  def arch_parameters(self):
    return self._arch_parameters

  # fixes architecture
  def fix_architecture(self, switch, new_weights = None):
    self._architecture_fixed = switch
    if new_weights is not None:
      self.alphas_normal = new_weights
    return

  def sample_alphas_normal(self, sample_amp=1, fair_darts_weight_threshold=0):

    alphas_normal = self.alphas_normal.clone()
    alphas_normal_sample = Variable(torch.zeros(alphas_normal.data.shape))

    for edge in range(alphas_normal.data.shape[0]):
      if self.DARTS_type == DARTS_Type.ORIGINAL:
        W_soft = F.softmax(alphas_normal[edge] * sample_amp, dim=0)
      elif self.DARTS_type == DARTS_Type.FAIR:
        transformed_alphas_normal = alphas_normal[edge]
        above_threshold = False
        for idx in range(len(transformed_alphas_normal.data)):
            if torch.sigmoid(transformed_alphas_normal).data[idx] > fair_darts_weight_threshold:
              above_threshold = True
              break
        if above_threshold:
          W_soft = F.softmax(transformed_alphas_normal * sample_amp, dim=0)
        else:
          W_soft = Variable(torch.zeros(alphas_normal[edge].shape))
          W_soft[PRIMITIVES.index('none')] = 1

      else:
        raise Exception("DARTS Type " + str(self.DARTS_type) + " not implemented")

      if torch.any(W_soft != W_soft):
        warnings.warn('Cannot properly sample from architecture weights due to nan entries.')
        k_sample = random.randrange(len(W_soft))
      else:
        k_sample = np.random.choice(range(len(W_soft)), p=W_soft.data.numpy())
      alphas_normal_sample[edge, k_sample] = 1

    return alphas_normal_sample

  def max_alphas_normal(self):

    alphas_normal = self.alphas_normal.clone()
    alphas_normal_sample = Variable(torch.zeros(alphas_normal.data.shape))

    for edge in range(alphas_normal.data.shape[0]):
      row = alphas_normal[edge]
      max_idx = np.argmax(row.data)
      alphas_normal_sample[edge, max_idx] = 1

    return alphas_normal_sample

  # returns the genotype of the model
  def genotype(self, sample = False):

    # this function uses the architecture weights to retrieve the operations with the highest weights
    def _parse(weights):
      gene = []
      n = self._n_input_states # 2 ... changed this to adapt to number of input states
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        # first get all the edges for a given node, edges are sorted according to their highest (non-none) weight, starting from the edge with the smallest heighest weight
        edges = sorted(range(n), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))
        # for each edge, figure out which is the primitive with the highest
        for j in edges:    # looping through all the edges for the current node (i)
          if sample:
            W_soft = F.softmax(Variable(torch.from_numpy(W[j])))
            k_best = np.random.choice(range(len(W[j])), p = W_soft.data.numpy())
          else:
            k_best = None
            for k in range(len(W[j])): # looping through all the primitives
              # choose the primitive with the highest weight
              # if k != PRIMITIVES.index('none'): # EDIT SM 01/13: commented to include "none" weights in genotype
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
            # add gene (primitive, edge number)
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    if self._architecture_fixed:
      gene_normal = _parse(self.alphas_normal.data.cpu().numpy())
    else:
      gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
    )
    return genotype

  def countParameters(self, print_parameters=False):
    n_params_total = 0 # counts only parameters of operations with the highest architecture weight

    # count classifier
    for parameter in self.classifier.parameters():
      if(parameter.requires_grad == True):
        n_params_total += parameter.data.numel()

    # count stem
    for parameter in self.stem.parameters():
      if(parameter.requires_grad == True):
        n_params_total += parameter.data.numel()

    n_params_base = n_params_total # number of parameters, excluding individual cells

    param_list = list()
    # now count number of parameters for cells that have highest probability
    for idx, op in enumerate(self.cells._ops):
      # pick most operation with highest likelihood
      values = self.alphas_normal[idx, :].data.numpy()
      maxIdx = np.where(values == max(values))

      tmp_param_list = list()
      if Network.isiterable(op._ops[np.asscalar(maxIdx[0])]): # Zero is not iterable

        for subop in op._ops[np.asscalar(maxIdx[0])]:

          for parameter in subop.parameters():
            tmp_param_list.append(parameter.data.numpy().squeeze())
            if (parameter.requires_grad == True):
              n_params_total += parameter.data.numel()

      print_parameters = True
      if print_parameters:
        print('Edge (' + str(idx) + '): ' + get_operation_label(PRIMITIVES[np.asscalar(maxIdx[0])], tmp_param_list))
      param_list.append(tmp_param_list)

    # # get parameters from final linear classifier
    # tmp_param_list = list()
    # for parameter in self.classifier.parameters():
    #   for subparameter in parameter:
    #     tmp_param_list.append(subparameter.data.numpy().squeeze())

    # get parameters from final linear for each edge
    for edge in range(self._steps):
      tmp_param_list = list()
      # add weight
      tmp_param_list.append(self.classifier._parameters['weight'].data[:,edge].numpy())
      # add partial bias (bias of classifier units will be devided by number of edges)
      if 'bias' in self.classifier._parameters.keys() and edge == 0:
        tmp_param_list.append(self.classifier._parameters['bias'].data.numpy())
      param_list.append(tmp_param_list)

      if print_parameters:
        print('Classifier from Edge ' + str(edge) + ': ' + get_operation_label('classifier_concat', tmp_param_list))

    return (n_params_total, n_params_base, param_list)

  def isiterable(p_object):
    try:
      it = iter(p_object)
    except TypeError:
      return False
    return True