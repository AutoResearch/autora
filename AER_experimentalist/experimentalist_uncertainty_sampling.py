import numpy as np
import torch

from sweetpea.primitives import Factor
from sweetpea import fully_cross_block, synthesize_trials_non_uniform

from abc import ABC, abstractmethod
from AER_experimentalist.experimentalist import Experimentalist

from alipy.query_strategy.query_labels import QueryInstanceUncertainty


# closed cycle from run_stroop_study.py
# AER_cycles = 20
# for cycle in range(AER_cycles):
#     print('CURRENT AER CYCLE:', cycle)

#     model = theorist.search_model(study_object, cycle+1)

#     experiment_file_path = experimentalist.sample_experiment(model, study_object)
#     data = experimentalist.commission_experiment(study_object, experiment_file_path)
#     study_object.add_data(data) 

# print('I am done !')


class Experimentalist_Uncertainty_Sampling(Experimentalist, ABC):

    def __init__(self, study_name, experiment_server_host=None, experiment_server_port=None, seed_data_file="", experiment_design=None, ivs=None):
        super().__init__(study_name, experiment_server_host, experiment_server_port, seed_data_file, experiment_design, ivs)
        self._query = QueryInstanceUncertainty(measure='least_confident')

        # uniformly sample the pool of samples
        experiment_design = list()
        resolution = 5 # hard coded to match self._seed_parameters[0] in experimentalist.py
        for var in ivs:
            factor = Factor(var.get_name(),
                            np.linspace(var._value_range[0], var._value_range[1], resolution).tolist())
            experiment_design.append(factor)

        block = fully_cross_block(experiment_design, experiment_design, []) # generate crossed experiment with SweetPea
        experiment_sequence = synthesize_trials_non_uniform(block, 1)[0]

        # prepare pool of samples for model
        sample = []
        num_samples = len(experiment_sequence[list(experiment_sequence.keys())[0]])
        for i in range(num_samples):
            cond = []
            for key in experiment_sequence:
                cond.append(float(experiment_sequence[key][i]))
            sample.append(cond)

        self.input_data = torch.Tensor(sample)
        self.indices = list(range(num_samples))  
        self.keys = list(experiment_sequence.keys())

    def init_experiment_search(self, model, object_of_study):
        super().init_experiment_search(model, object_of_study)
        return

    def sample_experiment_condition(self, model, object_of_study, condition):
        # input_data == torch tensor ([# of stuff, 5])
        # output_data == torch tensor ([# of stuff, 2])
        # model_prediction = model(input_data) # .forward using __call__

        # get model predictions
        output_data = model(self.input_data)
        predictions = np.empty(tuple(output_data.shape))

        for i in range(output_data.shape[0]):
            predictions[i] = output_data[i].detach().numpy()

        # sample from pool using uncertainty of model predictions
        query_indices = self._query.select_by_prediction_mat(self.indices, predictions, batch_size=(condition+1)) # indices of samples selected
        
        # write sampled conditions to a dict
        condition_data = self.input_data[query_indices[-1]]

        condition = {}
        for j in range(len(self.keys)):
            condition[self.keys[j]] = float(condition_data[j])

        return condition
