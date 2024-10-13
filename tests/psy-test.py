from psyneulink import *
from graph_scheduler import EveryNCalls
import numpy as np

# Construct the color naming pathway:
color_input = ProcessingMechanism(name='COLOR INPUT', size=2)  # note: default function is Linear
color_input_to_hidden_wts = np.array([[2, -2], [-2, 2]])
color_hidden = ProcessingMechanism(name='COLOR HIDDEN', size=2, function=Logistic(bias=-4))
color_hidden_to_output_wts = np.array([[2, -2], [-2, 2]])
output = ProcessingMechanism(name='OUTPUT', size=2, function=Logistic)
color_pathway = [color_input, color_input_to_hidden_wts, color_hidden, color_hidden_to_output_wts,
                 output]

# Construct the word reading pathway (using the same output_layer)
word_input = ProcessingMechanism(name='WORD INPUT', size=2)
word_input_to_hidden_wts = np.array([[3, -3], [-3, 3]])
word_hidden = ProcessingMechanism(name='WORD HIDDEN', size=2, function=Logistic(bias=-4))
word_hidden_to_output_wts = np.array([[3, -3], [-3, 3]])
word_pathway = [word_input, word_input_to_hidden_wts, word_hidden, word_hidden_to_output_wts,
                output]

# Construct the task specification pathways
task_input = ProcessingMechanism(name='TASK INPUT', size=2)
task = LCAMechanism(name='TASK', size=2)
task_color_wts = np.array([[4, 4], [0, 0]])
task_word_wts = np.array([[0, 0], [4, 4]])
task_color_pathway = [task_input, task, task_color_wts, color_hidden]
task_word_pathway = [task_input, task, task_word_wts, word_hidden]

# Construct the decision pathway:
decision = DDM(name='DECISION',
               input_format=ARRAY,
               # reset_stateful_function_when=AtTrialStart(),
               # function=DriftDiffusionIntegrator(noise=1., threshold=10)
               )
decision_pathway = [output, decision]

control = ControlMechanism(name='CONTROL',
                           objective_mechanism=ObjectiveMechanism(name='Conflict Monitor',
                                                                  monitor=output,
                                                                  function=Energy(size=2,
                                                                                  matrix=[[0, -3.],
                                                                                          [-3.,
                                                                                           0]])),
                           default_allocation=[0.5],
                           control_signals=[(GAIN, task)])

# Construct the Composition:
Stroop_model = Composition(name='Stroop Model')
Stroop_model.add_linear_processing_pathway(color_pathway)
Stroop_model.add_linear_processing_pathway(word_pathway)
Stroop_model.add_linear_processing_pathway(task_color_pathway)
Stroop_model.add_linear_processing_pathway(task_word_pathway)
Stroop_model.add_linear_processing_pathway(decision_pathway)
Stroop_model.add_controller(control)


# Stroop_model.scheduler.add_condition(color_hidden, EveryNCalls(task, 100))
# Stroop_model.scheduler.add_condition(word_hidden, EveryNCalls(task, 100))
# Stroop_model.scheduler.add_condition(output,All(EveryNCalls(color_hidden, 1),
#                                                            EveryNCalls(word_hidden, 1)))
# Stroop_model.scheduler.add_condition(decision, EveryNCalls(output, 1))

def converge(node, thresh, context):
    for val in node.parameters.value.get_delta(context):
        if any(abs(val) >= thresh):
            return False
    return True

#
epsilon = 0.001
Stroop_model.scheduler.add_condition(color_hidden, (Condition(converge, task, epsilon)))
Stroop_model.scheduler.add_condition(word_hidden, (Condition(converge, task, epsilon)))

red = [1, 0]
green = [0, 1]
word = [0, 1]
color = [1, 0]
# Trial 1  Trial 2
np.set_printoptions(precision=2)
global t
t = 0

def print_after():
    global t
    print(f'\nEnd of trial {t}:')
    print(f'\t\t\t\tcolor  word')
    print(f'\ttask:\t\t{task.value[0]}')
    print(f'\ttask gain:\t   {task.parameter_ports[GAIN].value}')
    print(f'\t\t\t\tred   green')
    print(f'\toutput:\t\t{output.value[0]}')
    print(f'\tdecision:\t{decision.value[0]}{decision.value[1]}')
    print(f'\tconflict:\t  {control.objective_mechanism.value[0]}')
    t += 1


# Set up run and then execute it
task.initial_value = [0.5, 0.5]  # Assign "neutral" starting point for task units on each trial
task.reset_stateful_function_when = AtTrialStart()  # Reset task units at beginning of each trial


num_trials = 4
stimuli = {color_input: [red] * num_trials + [green] * 1 + [red] * 2,
           word_input: [green] * num_trials + [green] * 1 + [green] * 2,
           task_input: [color] * num_trials + [color] * 1 + [color] *2}
Stroop_model.run(inputs=stimuli, call_after_trial=print_after)

# [[array([1.]), array([2.80488344])], [array([1.]), array([3.94471513])]]
