from ._state import State, StateMachine, Transition


class _CycleOwnStateMachine:
    def __init__(self, theorist, experimentalist, experiment_runner):
        """

        Args:
            theorist:
            experimentalist:
            experiment_runner:

        Examples:
            >>> from autora.skl.darts import DARTSRegressor
            >>> from autora.variable import VariableCollection, Variable
            >>> metadata = VariableCollection(
            ...    independent_variables=[Variable(name="x1", value_range=(-5, 5))],
            ...    dependent_variables=[Variable(name="y", value_range=(-10, 10))],
            ...    )
            >>> example_theorist = DARTSRegressor(num_graph_nodes=2, param_updates_per_epoch=1,
            ...     max_epochs=100, primitives=("none", "linear", "add", "subtract"))
            >>> example_experimentalist = make_pipeline([])
            >>> _CycleOwnStateMachine(theorist=example_theorist,
            ...     experimentalist=None, \
            ...     experiment_runner=None)
        """

        machine = StateMachine()

        state_start = State("start")
        state_theorist = State("theorist", callback=theorist)
        state_experimentalist = State("experimentalist", callback=experimentalist)
        state_experiment_runner = State("experiment runner", callback=experiment_runner)
        state_end = State("end")

        machine.states.append(state_start)
        machine.states.append(state_theorist)
        machine.states.append(state_experimentalist)
        machine.states.append(state_experiment_runner)
        machine.states.append(state_end)

        transition_start_experimentalist = Transition(
            state1=state_start, state2=state_experimentalist
        )
        transition_start_experiment_runner = Transition(
            state1=state_start, state2=state_experiment_runner
        )
        transition_start_theorist = Transition(
            state1=state_start, state2=state_theorist
        )
        machine.transitions.append(transition_start_experimentalist)
        machine.transitions.append(transition_start_experiment_runner)
        machine.transitions.append(transition_start_theorist)

        transition_theorist_experimentalist = Transition(
            state1=state_theorist, state2=state_experimentalist
        )
        transition_experimentalist_experiment_runner = Transition(
            state1=state_experimentalist,
            state2=state_experiment_runner,
        )
        transition_experiment_runner_theorist = Transition(
            state1=state_experiment_runner,
            state2=state_theorist,
        )
        machine.transitions.append(transition_theorist_experimentalist)
        machine.transitions.append(transition_experimentalist_experiment_runner)
        machine.transitions.append(transition_experiment_runner_theorist)

        transition_experimentalist_end = Transition(
            state1=state_experimentalist,
            state2=state_end,
        )
        transition_experiment_runner_end = Transition(
            state1=state_experiment_runner,
            state2=state_end,
        )
        transition_theorist_end = Transition(
            state1=state_theorist,
            state2=state_end,
        )
        machine.transitions.append(transition_experimentalist_end)
        machine.transitions.append(transition_experiment_runner_end)
        machine.transitions.append(transition_theorist_end)
