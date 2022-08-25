import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from autora.theorist.darts.model_search import DARTSType, Network
from autora.theorist.darts.operations import isiterable


def _concat(xs) -> torch.Tensor:
    """
    A function to concatenate a list of tensors.
    Args:
        xs: The list of tensors to concatenate.

    Returns:
        The concatenated tensor.
    """
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    """
    A learner operating on the architecture weights of a DARTS model.
    This learner handles training the weights associated with mixture operations
    (architecture weights).
    """

    def __init__(
        self,
        model: Network,
        arch_learning_rate_max: float,
        arch_momentum: float,
        arch_weight_decay: float,
        arch_weight_decay_df: float = 0,
        arch_weight_decay_base: float = 0,
        fair_darts_loss_weight: float = 1,
    ):
        """
        Initializes the architecture learner.

        Arguments:
            model: a network model implementing the full DARTS model.
            arch_learning_rate_max: learning rate for the architecture weights
            arch_momentum: arch_momentum used in the Adam optimizer for architecture weights
            arch_weight_decay: general weight decay for the architecture weights
            arch_weight_decay_df: (weight decay applied to architecture weights in proportion
                to the number of parameters of an operation)
            arch_weight_decay_base: (a constant weight decay applied to architecture weights)
            fair_darts_loss_weight: (a regularizer that pushes architecture weights more toward
                zero or one in the fair DARTS variant)
        """
        # set parameters for architecture learning
        self.network_arch_momentum = arch_momentum
        self.network_weight_decay = arch_weight_decay
        self.network_weight_decay_df = arch_weight_decay_df
        self.arch_weight_decay_base = arch_weight_decay_base * model._steps
        self.fair_darts_loss_weight = fair_darts_loss_weight

        self.model = model
        self.lr = arch_learning_rate_max
        # architecture is optimized using Adam
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=arch_learning_rate_max,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )

        # initialize weight decay matrix
        self._init_decay_weights()

        # initialize the logged loss
        self.current_loss = 0

    def _init_decay_weights(self):
        """
        This function initializes the weight decay matrix. The weight decay matrix
        is subtracted from the architecture weight matrix on every learning step. The matrix
        specifies a weight decay which is proportional to the number of parameters used in an
        operation.
        """
        n_params = list()
        for operation in self.model.cells._ops[0]._ops:
            if isiterable(operation):
                n_params_total = (
                    1  # any non-zero operation is counted as an additional parameter
                )
                for subop in operation:
                    for parameter in subop.parameters():
                        if parameter.requires_grad is True:
                            n_params_total += parameter.data.numel()
            else:
                n_params_total = 0  # no operation gets zero parameters
            n_params.append(n_params_total)

        self.decay_weights = Variable(
            torch.zeros(self.model.arch_parameters()[0].data.shape)
        )
        for idx, param in enumerate(n_params):
            if param > 0:
                self.decay_weights[:, idx] = (
                    param * self.network_weight_decay_df + self.arch_weight_decay_base
                )
            else:
                self.decay_weights[:, idx] = param
        self.decay_weights = self.decay_weights
        self.decay_weights = self.decay_weights.data

    def _compute_unrolled_model(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        eta: float,
        network_optimizer: torch.optim.Optimizer,
    ):
        """
        Helper function used to compute the approximate architecture gradient.

        Arguments:
            input: input patterns
            target: target patterns
            eta: learning rate
            network_optimizer: optimizer used to updating the architecture weights

        Returns:
            unrolled_model: the unrolled architecture
        """
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]["momentum_buffer"]
                for v in self.model.parameters()
            ).mul_(self.network_arch_momentum)
        except Exception:
            moment = torch.zeros_like(theta)
        dtheta = (
            _concat(torch.autograd.grad(loss, self.model.parameters())).data
            + self.network_weight_decay * theta
        )
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta)
        )
        return unrolled_model

    def step(
        self,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        network_optimizer: torch.optim.Optimizer,
        unrolled: bool,
        input_train: torch.Tensor = None,
        target_train: torch.Tensor = None,
        eta: float = 1,
    ):
        """
        Updates the architecture parameters for one training iteration

        Arguments:
            input_valid: input patterns for validation set
            target_valid: target patterns for validation set
            network_optimizer: optimizer used to updating the architecture weights
            unrolled: whether to use the unrolled architecture or not (i.e., whether to use
                the approximate architecture gradient or not)
            input_train: input patterns for training set
            target_train: target patterns for training set
            eta: learning rate for the architecture weights
        """

        # input_train, target_train only needed for approximation (unrolled=True)
        # of architecture gradient
        # when performing a single weigh update

        # initialize gradients to be zero
        self.optimizer.zero_grad()
        # use different backward step depending on whether to use
        # 2nd order approximation for gradient update
        if unrolled:  # probably using eta of parameter update here
            self._backward_step_unrolled(
                input_train,
                target_train,
                input_valid,
                target_valid,
                eta,
                network_optimizer,
            )
        else:
            self._backward_step(input_valid, target_valid)
        # move Adam one step
        self.optimizer.step()

    # backward step (using non-approximate architecture gradient, i.e., full training)
    def _backward_step(self, input_valid: torch.Tensor, target_valid: torch.Tensor):
        """
        Computes the loss and updates the architecture weights assuming full optimization
        of coefficients for the current architecture.

        Arguments:
            input_valid: input patterns for validation set
            target_valid: target patterns for validation set
        """
        if self.model.DARTS_type == DARTSType.ORIGINAL:
            loss = self.model._loss(input_valid, target_valid)
        elif self.model.DARTS_type == DARTSType.FAIR:
            loss1 = self.model._loss(input_valid, target_valid)
            loss2 = -F.mse_loss(
                torch.sigmoid(self.model.alphas_normal),
                0.5 * torch.ones(self.model.alphas_normal.shape, requires_grad=False),
            )  # torch.tensor(0.5, requires_grad=False)
            loss = loss1 + self.fair_darts_loss_weight * loss2
        else:
            raise Exception(
                "DARTS Type " + str(self.model.DARTS_type) + " not implemented"
            )

        loss.backward()
        self.current_loss = loss.item()

        # weight decay proportional to degrees of freedom
        for p in self.model.arch_parameters():
            p.data.sub_((self.decay_weights * self.lr))  # weight decay

    # backward pass using second order approximation
    def _backward_step_unrolled(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        eta: float,
        network_optimizer: torch.optim.Optimizer,
    ):
        """
        Computes the loss and updates the architecture weights using the approximate architecture
        gradient.

        Arguments:
            input_train: input patterns for training set
            target_train: target patterns for training set
            input_valid: input patterns for validation set
            target_valid: target patterns for validation set
            eta: learning rate
            network_optimizer: optimizer used to updating the architecture weights

        """

        # gets the model
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer
        )

        if self.model.DARTS_type == DARTSType.ORIGINAL:
            unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        elif self.model.DARTS_type == DARTSType.FAIR:
            loss1 = self.model._loss(input_valid, target_valid)
            loss2 = -F.mse_loss(
                torch.sigmoid(self.model.alphas_normal),
                torch.tensor(0.5, requires_grad=False),
            )
            unrolled_loss = loss1 + self.fair_darts_loss_weight * loss2
        else:
            raise Exception(
                "DARTS Type " + str(self.model.DARTS_type) + " not implemented"
            )

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

    def _construct_model_from_theta(self, theta: torch.Tensor):
        """
        Helper function used to compute the approximate gradient update
        for the architecture weights.

        Arguments:
            theta: term used to compute approximate gradient update

        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : (offset + v_length)].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new  # .cuda() # Edit SM 10/26/19: uncommented for cuda

    # second order approximation of architecture gradient (see Eqn. 8 from Liu et al, 2019)
    def _hessian_vector_product(
        self, vector: torch.Tensor, input: torch.Tensor, target: torch.Tensor, r=1e-2
    ):
        """
        Helper function used to compute the approximate gradient update
        for the architecture weights. It computes the hessian vector product outlined in Eqn. 8
        from Liu et al, 2019.

        Arguments:
            vector: input vector
            input: input patterns
            target: target patterns
            r: coefficient used to compute the hessian vector product

        """
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        # this implements Eqn. 8 from Liu et al. (2019)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
