import random
import warnings
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from aer.theorist.darts.fan_out import Fan_Out
from aer.theorist.darts.genotypes import PRIMITIVES, Genotype
from aer.theorist.darts.operations import OPS, get_operation_label


class DARTS_Type(Enum):
    """
    Enumerator class that indexes different variants of DARTS.
    """

    ORIGINAL = (
        # Liu, Simonyan & Yang (2018). Darts: Differentiable architecture
        # search
        1
    )
    # Chu, Zhou, Zhang & Li (2020). Fair darts: Eliminating unfair advantages
    # in differentiable architecture search
    FAIR = 2


# for 2 input nodes, 1 output node and 4 intermediate nodes,
# there are 14 possible edges (x 8 operations)
# Let input nodes be 1, 2 intermediate nodes 3, 4, 5, 6, and output node 7
# The edges are 3-1, 3-2; 4-1, 4-2, 4-3; 5-1, 5-2, 5-3, 5-4; 6-1, 6-2,
# 6-3, 6-4, 6-5; 2 + 3 + 4 + 5 = 14 edges


class MixedOp(nn.Module):
    """
    Mixture operation as applied in Differentiable Architecture Search (DARTS).
    A mixture operation amounts to a weighted mixture of a pre-defined set of operations
    that is applied to an input variable.
    """

    def __init__(self):
        """
        Initializes a mixture operation based on a pre-specified set of primitive operations.
        """
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        # loop through all the 8 primitive operations
        for primitive in PRIMITIVES:
            # define operation
            # OPS returns an nn module for a given primitive (defines as a string)
            op = OPS[primitive](False)

            # add the operation
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Computes a mixture operation as a weighted sum of all primitive operations.
        Arguments:
            x: input to the mixture operations
            weights: weight vector containing the weights associated with each operation
        Returns:
            y: result of the weighted mixture operation
        """
        # there are 8 weights for all the eight primitives. then it returns the
        # weighted sum of all operations performed on a given input
        return sum(w * op(x) for w, op in zip(weights, self._ops))


# Let a cell be a DAG(directed acyclic graph) containing N nodes (2 input
# nodes 1 output node?)
class Cell(nn.Module):
    """
    A cell as defined in differentiable architecture search. A single cell corresponds
    to a computation graph with the number of input nodes defined by n_input_states and
    the number of hidden nodes defined by steps. Input nodes only project to hidden nodes and hidden
    nodes project to each other with an acyclic connectivity pattern. The output of a cell
    corresponds to the concatenation of all hidden nodes. Hidden nodes are computed by integrating
    transformed outputs from sending nodes. Outputs from sending nodes correspond to
    mixture operations, i.e. a weighted combination of pre-specified operations applied to the
    variable specified by the sending node (see MixedOp).

    Attributes:
        _steps: number of hidden nodes
        _n_input_states: number of input nodes
        _ops: list of mixture operations (amounts to the list of edges in the cell)
    """

    def __init__(self, steps, n_input_states):
        """
        Initializes a cell based on the number of hidden nodes (steps)
        and the number of input nodes (n_input_states).

        Arguments:
            steps: number of hidden nodes
            n_input_states: number of input nodes
        """
        # The first and second nodes of cell k are set equal to the outputs of
        # cell k − 2 and cell k − 1, respectively, and 1 × 1 convolutions
        # (ReLUConvBN) are inserted as necessary
        super(Cell, self).__init__()

        # set parameters
        self._steps = steps  # hidden nodes
        self._n_input_states = n_input_states  # input nodes

        # EDIT 11/04/19 SM: adapting to new SimpleNet data (changed from
        # multiplier to steps)
        self._multiplier = steps

        # set operations according to number of modules (empty)
        self._ops = nn.ModuleList()
        # iterate over edges: edges between each hidden node and input nodes +
        # prev hidden nodes
        for i in range(self._steps):  # hidden nodes
            for j in range(self._n_input_states + i):  # 2 refers to the 2 input nodes
                # defines the stride for link between cells
                # adds a mixed operation (derived from architecture parameters alpha)
                # for 4 intermediate nodes, a total of 14 connections
                # (MixedOps) is added
                op = MixedOp()
                # appends cell with mixed operation
                self._ops.append(op)

    def forward(self, input_states, weights):
        """
        Computes the output of a cell given a list of input states
        (variables represented in input nodes) and a weight matrix specifying the weights of each
        operation for each edge.

        Arguments:
            input_states: list of input nodes
            weights: matrix specifying architecture weights, i.e. the weights associated
            with each operation for each edge
        """
        # initialize states (activities of each node in the cell)
        states = list()

        # add each input node to the number of states
        for input in input_states:
            states.append(input)

        offset = 0
        # this computes the states from intermediate nodes and adds them to the list of states
        # (values of nodes)
        # for each hidden node, compute edge between existing states (input
        # nodes / previous hidden) nodes and current node
        for i in range(
            self._steps
        ):  # compute the state for each hidden node, first hidden node is
            # sum of input nodes, second is sum of input and first hidden
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        # concatenates the states of the last n (self._multiplier) intermediate
        # nodes to get the output of a cell
        result = torch.cat(states[-self._multiplier :], dim=1)
        return result


class Network(nn.Module):
    """
    A PyTorch computation graph according to DARTS.
    It consists of a single computation cell which transforms an
    input vector (containing all input variable) into an output vector, by applying a set of
    mixture operations which are defined by the architecture weights (labeled "alphas" of the
    network).

    The network flow looks as follows: An input vector (with _n_input_states elements) is split into
    _n_input_states separate input nodes (one node per element). The input nodes are then passed
    through a computation cell with _steps hidden nodes (see Cell). The output of the computation
    cell corresponds to the concatenation of its hidden nodes (a single vector). The final output
    corresponds to a (trained) affine transformation of this concatenation (labeled "classifier").

    Attributes:
        _n_input_states: length of input vector (translates to number of input nodes)
        _num_classes: length of output vector
        _criterion: optimization criterion used to define the loss
        _steps: number of hidden nodes in the cell
        _architecture_fixed: specifies whether the architecture weights shall remain fixed
        (not trained)
        _classifier_weight_decay: a weight decay applied to the classifier
        _stem: an operation for splitting the input vector into _n_input_states input nodes
    """

    def __init__(
        self,
        num_classes,
        criterion,
        steps=2,
        n_input_states=2,
        architecture_fixed=False,
        classifier_weight_decay=0,
        darts_type=DARTS_Type.ORIGINAL,
    ):
        """
        Initializes the network.

        Arguments:
            num_classes: length of output vector
            criterion: optimization criterion used to define the loss
            steps: number of hidden nodes in the cell
            n_input_states: length of input vector (translates to number of input nodes)
            architecture_fixed: specifies whether the architecture weights shall remain fixed
            classifier_weight_decay: a weight decay applied to the classifier
            darts_type: variant of DARTS (regular or fair) that is applied for training
        """
        super(Network, self).__init__()
        self.DARTS_type = darts_type
        # set parameters
        self._num_classes = num_classes  # number of output classes
        self._criterion = criterion  # optimization criterion (e.g. softmax)
        self._steps = steps  # the number of intermediate nodes (4)
        self._n_input_states = n_input_states  # number of input nodes
        self._multiplier = (
            1  # the number of internal nodes that get concatenated to the output
        )
        self._dim_output = self._steps
        self._architecture_fixed = architecture_fixed
        self._classifier_weight_decay = classifier_weight_decay

        # input nodes
        self.stem = nn.Sequential(Fan_Out(self._n_input_states))

        self.cells = (
            nn.ModuleList()
        )  # get list of all current modules (should be empty)

        # generate a cell that undergoes architecture search
        self.cells = Cell(steps, self._n_input_states)

        # last layer is a linear classifier (e.g. with 10 CIFAR classes)
        self.classifier = nn.Linear(
            self._dim_output, num_classes
        )  # make this the number of input states

        # initializes weights of the architecture
        self._initialize_alphas()

    # function for copying the network
    def new(self):
        model_new = Network(
            self._C, self._num_classes, self._criterion, steps=self._steps
        )
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    # computes forward pass for full network
    def forward(self, input):
        """
        Computes output of the network.

        Arguments:
            input: input to the network
        """

        # compute stem first
        input_states = self.stem(input)

        # get architecture weights
        if self._architecture_fixed:
            weights = self.alphas_normal
        else:
            if self.DARTS_type == DARTS_Type.ORIGINAL:
                weights = F.softmax(self.alphas_normal, dim=-1)
            elif self.DARTS_type == DARTS_Type.FAIR:
                weights = torch.sigmoid(self.alphas_normal)
            else:
                raise Exception(
                    "DARTS Type " + str(self.DARTS_type) + " not implemented"
                )

        # then apply cell with weights
        cell_output = self.cells(input_states, weights)

        # compute logits
        logits = self.classifier(cell_output.view(cell_output.size(0), -1))
        # just gets output to have only 2 dimensions (batch_size x num units in
        # output layer)

        return logits

    def _loss(self, input, target):
        """
        Computes the loss of the network for the specified criterion.

        Arguments:
            input: input patterns
            target: target patterns
        Returns: loss
        """
        logits = self(input)
        return self._criterion(logits, target)  # returns cross entropy by default

    # regularization
    def apply_weight_decay_to_classifier(self, lr):
        """
        Applies a weight decay to the weights projecting from the cell to the final output layer.

        Arguments:
            lr: learning rate
        """
        # weight decay proportional to degrees of freedom
        for p in self.classifier.parameters():
            p.data.sub_(
                self._classifier_weight_decay
                * lr
                * torch.sign(p.data)
                * (torch.abs(p.data))
            )  # weight decay

    def _initialize_alphas(self):
        """
        Initializes the architecture weights.
        """
        # compute the number of possible connections between nodes
        k = sum(1 for i in range(self._steps) for n in range(self._n_input_states + i))
        # number of available primitive operations (8 different types for a
        # conv net)
        num_ops = len(PRIMITIVES)

        # e.g., generate 14 (number of available edges) by 8 (operations)
        # weight matrix for normal alphas of the architecture
        self.alphas_normal = Variable(
            1e-3 * torch.randn(k, num_ops), requires_grad=True
        )
        # those are all the parameters of the architecture
        self._arch_parameters = [self.alphas_normal]

    # provide back the architecture as a parameter
    def arch_parameters(self):
        """
        Returns architecture weights.

        Returns:
            self._arch_parameters: architecture weights.
        """
        return self._arch_parameters

    # fixes architecture
    def fix_architecture(self, switch: bool, new_weights=None):
        """
        Freezes or unfreezes the architecture weights.

        Arguments:
            switch: set true to freeze architecture weights or false unfreeze
            new_weights: new set of architecture weights
        """
        self._architecture_fixed = switch
        if new_weights is not None:
            self.alphas_normal = new_weights
        return

    def sample_alphas_normal(self, sample_amp=1, fair_darts_weight_threshold=0):
        """
        Samples an architecture from the mixed operations from a probability distribution that is
        defined by the (softmaxed) architecture weights.
        This amounts to selecting one operation per edge (i.e., setting the architecture
        weight of that operation to one while setting the others to zero).

        Arguments:
            sample_amp: temperature that is applied before passing the weights through a softmax
            fair_darts_weight_threshold: used in fair DARTS. If an architecture weight is below
            this value then it is set to zero.

        Returns:
            alphas_normal_sample: sampled architecture weights.
        """

        alphas_normal = self.alphas_normal.clone()
        alphas_normal_sample = Variable(torch.zeros(alphas_normal.data.shape))

        for edge in range(alphas_normal.data.shape[0]):
            if self.DARTS_type == DARTS_Type.ORIGINAL:
                W_soft = F.softmax(alphas_normal[edge] * sample_amp, dim=0)
            elif self.DARTS_type == DARTS_Type.FAIR:
                transformed_alphas_normal = alphas_normal[edge]
                above_threshold = False
                for idx in range(len(transformed_alphas_normal.data)):
                    if (
                        torch.sigmoid(transformed_alphas_normal).data[idx]
                        > fair_darts_weight_threshold
                    ):
                        above_threshold = True
                        break
                if above_threshold:
                    W_soft = F.softmax(transformed_alphas_normal * sample_amp, dim=0)
                else:
                    W_soft = Variable(torch.zeros(alphas_normal[edge].shape))
                    W_soft[PRIMITIVES.index("none")] = 1

            else:
                raise Exception(
                    "DARTS Type " + str(self.DARTS_type) + " not implemented"
                )

            if torch.any(W_soft != W_soft):
                warnings.warn(
                    "Cannot properly sample from architecture weights due to nan entries."
                )
                k_sample = random.randrange(len(W_soft))
            else:
                k_sample = np.random.choice(range(len(W_soft)), p=W_soft.data.numpy())
            alphas_normal_sample[edge, k_sample] = 1

        return alphas_normal_sample

    def max_alphas_normal(self):
        """
        Samples an architecture from the mixed operations by selecting, for each edge,
        the operation with the largest architecture weight.

        Returns:
            alphas_normal_sample: sampled architecture weights.
        """
        alphas_normal = self.alphas_normal.clone()
        alphas_normal_sample = Variable(torch.zeros(alphas_normal.data.shape))

        for edge in range(alphas_normal.data.shape[0]):
            row = alphas_normal[edge]
            max_idx = np.argmax(row.data)
            alphas_normal_sample[edge, max_idx] = 1

        return alphas_normal_sample

    # returns the genotype of the model
    def genotype(self, sample=False):
        """
        Computes a genotype of the model which specifies the current computation graph based on
        the largest architecture weight for each edge, or based on a sample.
        The genotype can be used for parsing or plotting the computation graph.

        Arguments:
            sample: if set to true, the architecture will be determined by sampling
            from a probability distribution that is determined by the
            softmaxed architecture weights. If set to false (default), the architecture will be
            determined based on the largest architecture weight per edge.

        Returns:
            genotype: genotype describing the current (sampled) architecture
        """
        # this function uses the architecture weights to retrieve the
        # operations with the highest weights
        def _parse(weights):
            gene = []
            n = (
                self._n_input_states
            )  # 2 ... changed this to adapt to number of input states
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # first get all the edges for a given node, edges are sorted according to their
                # highest (non-none) weight, starting from the edge with the smallest heighest
                # weight
                edges = sorted(
                    range(n),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))
                        if k != PRIMITIVES.index("none")
                    ),
                )
                # for each edge, figure out which is the primitive with the
                # highest
                for (
                    j
                ) in edges:  # looping through all the edges for the current node (i)
                    if sample:
                        W_soft = F.softmax(Variable(torch.from_numpy(W[j])))
                        k_best = np.random.choice(
                            range(len(W[j])), p=W_soft.data.numpy()
                        )
                    else:
                        k_best = None
                        # looping through all the primitives
                        for k in range(len(W[j])):
                            # choose the primitive with the highest weight
                            # if k != PRIMITIVES.index('none'):
                            # EDIT SM 01/13: commented to include "none"
                            # weights in genotype
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
            gene_normal = _parse(
                F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
            )

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
        )
        return genotype

    def countParameters(self, print_parameters=False):
        """
        Counts and returns the parameters (coefficients) of the architecture defined by the
        highest architecture weights.

        Arguments:
            print_parameters: if set to true, the function will print all parameters.

        Returns:
            n_params_total: total number of parameters
            n_params_base: number of parameters determined by the classifier
            param_list: list of parameters specifying the corresponding edge (operation)
            and value
        """

        # counts only parameters of operations with the highest architecture weight
        n_params_total = 0

        # count classifier
        for parameter in self.classifier.parameters():
            if parameter.requires_grad is True:
                n_params_total += parameter.data.numel()

        # count stem
        for parameter in self.stem.parameters():
            if parameter.requires_grad is True:
                n_params_total += parameter.data.numel()

        n_params_base = (
            n_params_total  # number of parameters, excluding individual cells
        )

        param_list = list()
        # now count number of parameters for cells that have highest
        # probability
        for idx, op in enumerate(self.cells._ops):
            # pick most operation with highest likelihood
            values = self.alphas_normal[idx, :].data.numpy()
            maxIdx = np.where(values == max(values))

            tmp_param_list = list()
            if Network.isiterable(op._ops[maxIdx[0].item()]):  # Zero is not iterable

                for subop in op._ops[maxIdx[0].item()]:

                    for parameter in subop.parameters():
                        tmp_param_list.append(parameter.data.numpy().squeeze())
                        if parameter.requires_grad is True:
                            n_params_total += parameter.data.numel()

            print_parameters = True
            if print_parameters:
                print(
                    "Edge ("
                    + str(idx)
                    + "): "
                    + get_operation_label(PRIMITIVES[maxIdx[0].item()], tmp_param_list)
                )
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
            tmp_param_list.append(
                self.classifier._parameters["weight"].data[:, edge].numpy()
            )
            # add partial bias (bias of classifier units will be devided by
            # number of edges)
            if "bias" in self.classifier._parameters.keys() and edge == 0:
                tmp_param_list.append(self.classifier._parameters["bias"].data.numpy())
            param_list.append(tmp_param_list)

            if print_parameters:
                print(
                    "Classifier from Edge "
                    + str(edge)
                    + ": "
                    + get_operation_label("classifier_concat", tmp_param_list)
                )

        return (n_params_total, n_params_base, param_list)

    def isiterable(p_object):
        """
        A helper function that determines whether an object is iterable.

        Arguments:
            p_object: an object that is evaluated to be iterable or not

        Returns:
            is_iterable: true of the object is iterable
        """
        try:
            iter(p_object)
        except TypeError:
            return False
        return True
