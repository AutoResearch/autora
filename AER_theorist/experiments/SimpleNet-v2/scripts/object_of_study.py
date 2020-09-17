from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from enum import Enum

class outputTypes(Enum):
    REAL = 1
    PROBABILITY = 2
    PROBABILITY_DISTRIBUTION = 3
    CLASS = 4


class objectOfStudy(Dataset, ABC):

    def __init__(self, num_patterns=100, sampling=False):
        self._num_patterns = num_patterns
        self._sampling = sampling

    @abstractmethod
    def __get_input_dim__(self):
        pass

    @abstractmethod
    def __get_output_dim__(self):
        pass

    @abstractmethod
    def __get_output_type__(self):
        pass

    @abstractmethod
    def __get_name__(self):
        pass

    @abstractmethod
    def sample_model_fit(self):
        pass

    @abstractmethod
    def sample_model_fit_2d(self):
        pass




