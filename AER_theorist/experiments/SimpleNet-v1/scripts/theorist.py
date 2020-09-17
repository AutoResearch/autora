import AER_config as AER_cfg
from AER_theorist.object_of_study import Object_Of_Study
from enum import Enum

class architecture_search_strategy(Enum):
    DARTS = 1

class Theorist():

    object_of_study = None
    _architecture_search_strategy = None


    def __init__(self, object_of_study: Object_Of_Study, architecture_search_strategy=architecture_search_strategy.DARTS):

        self.object_of_study = Object_Of_Study
        self._architecture_search_strategy = architecture_search_strategy

class Theorist_DARTS(Theorist):

    def __init__(self, object_of_study: Object_Of_Study, cfg):

        super(Theorist_DARTS, self).__init__(object_of_study)

        # set configuration parameters from cfg

    def initialize_model_search(self):
        pass

    def commission_model_search(self):
        pass

    def run_model_search(self):
        pass


