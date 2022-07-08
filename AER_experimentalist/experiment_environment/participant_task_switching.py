 from AER_experimentalist.experiment_environment.participant_in_silico import Participant_In_Silico
import math
import torch
import numpy

class TaskSwitchModel():
   def __init__(self):
       self.noise = numpy.random.normal
       self.r = numpy.random.exponential(1/150.0) + numpy.random.normal(200, 240)
       self.P = 200
       self.R = 200

   def forward(self, input):
       input = torch.Tensor(input)
       if len(input.shape) <= 1:
           input = input.view(1, len(input))

       # convert inputs
       strength = torch.zeros(input.shape[0], 2)
       priming = torch.zeros(input.shape[0], 2)
       control = torch.zeros(input.shape[0], 2)

       strength[:, 0:2] = input[:, 0:2]
       priming[:, 0:2] = input[:, 2:4]
       control[:, 0:2] = input[:, 4:6]

       activation = torch.zeros(1, 2)
       total_activation = 0
       generation_times = torch.zeros(1, 2)
       output = torch.zeros(1, 2)

       for idx in enumerate(activation):
           inval = strength[0, idx] + priming[0, idx] + control[0, idx]
           activation[0, idx] = 1 - math.exp(-0.5 * inval)
           total_activation += activation[0, idx]

       for idx in enumerate(generation_times):
           generation_times[0, idx] = 100/(activation[0, idx]/total_activation)
           output[0, idx] = self.r + self.P + self.R + generation_times[0, idx]

       return output


class Participant_TaskSwitch(Participant_In_Silico):

   # initializes participant
   def __init__(self):
       super(Participant_TaskSwitch, self).__init__()

       self.strong_strength = torch.zeros(1, 1)
       self.weak_strength = torch.zeros(1, 1)
       self.strong_priming = torch.zeros(1, 1)
       self.weak_priming = torch.zeros(1, 1)
       self.strong_control = torch.ones(1, 1)  # color task units are activated by default
       self.weak_control = torch.zeros(1, 1)

       # read value from participant

   def get_value(self, variable_name):

       if variable_name is "strong_rt":
           return self.output[0, 0].numpy()

       elif variable_name is "weak_rt":
           return self.output[0, 1].numpy()

       raise Exception('Could not get value from Stroop Participant. Variable name "' + variable_name + '" not found.')

   def set_value(self, variable_name, value):

       if variable_name is "strong_strength":
           self.strong_strength[0, 0] = value

       elif variable_name is "weak_strength":
           self.weak_strength[0, 0] = value

       elif variable_name is "strong_priming":
           self.strong_priming[0, 0] = value

       elif variable_name is "weak_priming":
           self.weak_priming[0, 0] = value

       elif variable_name is "strong_control":
           self.strong_control[0, 0] = value

       elif variable_name is "task_word":
           self.weak_control[0, 0] = value

       else:
           raise Exception('Could not set value for Stroop Participant. Variable name "' + variable_name + '" not found.')

   def execute(self):

       input = torch.zeros(1, 6)
       input[0, 0] = self.strong_strength
       input[0, 1] = self.weak_strength
       input[0, 2] = self.strong_priming
       input[0, 3] = self.weak_priming
       input[0, 4] = self.strong_control
       input[0, 5] = self.weak_control

       # compute regular output
       self.output = self.model(input).detach()

   def run_exp(model):
       strong_strength = 0.2
       weak_strength = 0.2
       strong_priming = 0.2
       weak_priming = 0
       strong_control = 0.15
       weak_control = 0.15
       input = [strong_strength, weak_strength, strong_priming, weak_priming, strong_control, weak_control]
       output_eql_strong = torch.sigmoid(model(input))

       strong_strength = 0.2
       weak_strength = 0.2
       strong_priming = 0
       weak_priming = 0.2
       strong_control = 0.15
       weak_control = 0.15
       input = [strong_strength, weak_strength, strong_priming, weak_priming, strong_control, weak_control]
       output_eql_weak = torch.sigmoid(model(input))

       strong_strength = 0.2
       weak_strength = 0.7
       strong_priming = 0.2
       weak_priming = 0
       strong_control = 0.15
       weak_control = 0.15
       input = [strong_strength, weak_strength, strong_priming, weak_priming, strong_control, weak_control]
       output_uneql_strong = torch.sigmoid(model(input))

       strong_strength = 0.2
       weak_strength = 0.7
       strong_priming = 0
       weak_priming = 0.2
       strong_control = 0.15
       weak_control = 0.15
       input = [strong_strength, weak_strength, strong_priming, weak_priming, strong_control, weak_control]
       output_uneql_weak = torch.sigmoid(model(input))

       return (output_eql_strong, output_eql_weak, output_uneql_strong,
               output_uneql_weak)

