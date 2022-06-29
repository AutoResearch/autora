import torch

from aer.theorist.darts.model_search import DARTS_Type, Network
from aer.theorist.darts.utils import get_loss_function
from aer.variable import ValueType


class DARTS:
    def __init__(
        self,
        input_dimensions: int = 1,
        output_dimensions: int = 1,
        output_type: ValueType = ValueType("real"),
        darts_type: DARTS_Type = DARTS_Type("original"),
        max_iter: int = 100,
        arch_weight_decay_df: float = 3e-4,
        num_graph_nodes: int = 2,
        classifier_weight_decay: float = 1e-2,
        optimizer_weight_decay: float = 3e-4,
        learning_rate: float = 0.025,
        momentum: float = 0.9,
    ):

        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.output_type = output_type

        self.max_iter = max_iter
        self.darts_type = darts_type

        self.arch_weight_decay_df = arch_weight_decay_df
        self.classifier_weight_decay = classifier_weight_decay
        self.optimizer_weight_decay = optimizer_weight_decay
        self.num_graph_nodes = num_graph_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum

        pass

    def fit(self, X, y):

        # Do whatever happens in theorist.init_model_search
        self.model_ = Network(
            num_classes=self.output_dimensions,
            criterion=get_loss_function(ValueType(self.output_type)),
            steps=self.num_graph_nodes,
            n_input_states=self.input_dimensions,
            classifier_weight_decay=self.classifier_weight_decay,
            darts_type=DARTS_Type(self.darts_type),
        )

        self.optimizer_ = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.optimizer_weight_decay,
        )

        for n in range(self.max_iter):
            # Do whatever happens in theorist.run_model_search, n times
            pass

        return self

    def predict(self, X):
        pass
