"""

Functions and classes for running the complete AER cycle.

# Basic Usage

Aim: Use the Controller to recover a simple ground truth theory from noisy data.

Examples:

    >>> def ground_truth(x):
    ...     return x + 1

    The space of allowed x values is the integers between 0 and 10 inclusive,
    and we record the allowed output values as well.
    >>> from autora.variable import VariableCollection, Variable
    >>> metadata_0 = VariableCollection(
    ...    independent_variables=[Variable(name="x1", allowed_values=range(11))],
    ...    dependent_variables=[Variable(name="y", value_range=(-20, 20))],
    ...    )

    The experimentalist is used to propose experiments.
    Since the space of values is so restricted, we can just sample them all each time.
    >>> from autora.experimentalist.pipeline import make_pipeline
    >>> example_experimentalist = make_pipeline(
    ...     [metadata_0.independent_variables[0].allowed_values])

    When we run a synthetic experiment, we get a reproducible noisy result:
    >>> import numpy as np
    >>> def get_example_synthetic_experiment_runner():
    ...     rng = np.random.default_rng(seed=180)
    ...     def runner(x):
    ...         return ground_truth(x) + rng.normal(0, 0.1, x.shape)
    ...     return runner
    >>> example_synthetic_experiment_runner = get_example_synthetic_experiment_runner()
    >>> example_synthetic_experiment_runner(np.array([1]))
    array([2.04339546])

    The theorist "tries" to work out the best theory.
    We use a trivial scikit-learn regressor.
    >>> from sklearn.linear_model import LinearRegression
    >>> example_theorist = LinearRegression()

    We initialize the Controller with the metadata describing the domain of the theory,
    the theorist, experimentalist and experiment runner,
    as well as a monitor which will let us know which cycle we're currently on.
    >>> cycle = Cycle(
    ...     metadata=metadata_0,
    ...     theorist=example_theorist,
    ...     experimentalist=example_experimentalist,
    ...     experiment_runner=example_synthetic_experiment_runner,
    ...     monitor=lambda state: print(f"Generated {len(state.theories)} theories"),
    ... )
    >>> cycle # doctest: +ELLIPSIS
    <...Cycle object at 0x...>

    We can run the cycle by calling the run method:
    >>> cycle.run(num_cycles=3)  # doctest: +ELLIPSIS
    Generated 1 theories
    Generated 2 theories
    Generated 3 theories
    <...Cycle object at 0x...>

    We can now interrogate the results. The first set of conditions which went into the
    experiment runner were:
    >>> cycle.data.conditions[0]
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    The observations include the conditions and the results:
    >>> cycle.data.observations[0]
    array([[ 0.        ,  0.92675345],
           [ 1.        ,  1.89519928],
           [ 2.        ,  3.08746571],
           [ 3.        ,  3.93023943],
           [ 4.        ,  4.95429102],
           [ 5.        ,  6.04763988],
           [ 6.        ,  7.20770574],
           [ 7.        ,  7.85681519],
           [ 8.        ,  9.05735823],
           [ 9.        , 10.18713406],
           [10.        , 10.88517906]])

    In the third cycle (index = 2) the first and last values are different again:
    >>> cycle.data.observations[2][[0,-1]]
    array([[ 0.        ,  1.08559827],
           [10.        , 11.08179553]])

    The best fit theory after the first cycle is:
    >>> cycle.data.theories[0]
    LinearRegression()

    >>> def report_linear_fit(m: LinearRegression,  precision=4):
    ...     s = f"y = {np.round(m.coef_[0].item(), precision)} x " \\
    ...     f"+ {np.round(m.intercept_.item(), 4)}"
    ...     return s
    >>> report_linear_fit(cycle.data.theories[0])
    'y = 1.0089 x + 0.9589'

    The best fit theory after all the cycles, including all the data, is:
    >>> report_linear_fit(cycle.data.theories[-1])
    'y = 0.9989 x + 1.0292'

    This is close to the ground truth theory of x -> (x + 1)

    We can also run the cycle with more control over the execution flow:
    >>> next(cycle) # doctest: +ELLIPSIS
    Generated 4 theories
    <...Cycle object at 0x...>

    >>> next(cycle) # doctest: +ELLIPSIS
    Generated 5 theories
    <...Cycle object at 0x...>

    >>> next(cycle) # doctest: +ELLIPSIS
    Generated 6 theories
    <...Cycle object at 0x...>

    We can continue to run the cycle as long as we like,
    with a simple arbitrary stopping condition like the number of theories generated:
    >>> from itertools import takewhile
    >>> _ = list(takewhile(lambda c: len(c.data.theories) < 9, cycle))
    Generated 7 theories
    Generated 8 theories
    Generated 9 theories

    ... or the precision (here we keep iterating while the difference between the gradients
    of the second-last and last cycle is larger than 1x10^-3).
    >>> _ = list(
    ...         takewhile(
    ...             lambda c: np.abs(c.data.theories[-1].coef_.item() -
    ...                            c.data.theories[-2].coef_.item()) > 1e-3,
    ...             cycle
    ...         )
    ...     )
    Generated 10 theories
    Generated 11 theories

    ... or continue to run as long as we like:
    >>> _ = cycle.run(num_cycles=100) # doctest: +ELLIPSIS
    Generated 12 theories
    ...
    Generated 111 theories

# Passing Static Parameters

Aim: pass parameters to the cycle components, when they are needed.

Examples:

    Here we have an experimentalist which takes a parameter:
    >>> uniform_random_rng = np.random.default_rng(180)
    >>> def uniform_random_sampler(n):
    ...     return uniform_random_rng.uniform(low=0, high=11, size=n)
    >>> example_experimentalist_with_parameters = make_pipeline([uniform_random_sampler])

    The cycle can handle that using the `params` keyword:
    >>> cycle_with_parameters = Cycle(
    ...     metadata=metadata_0,
    ...     theorist=example_theorist,
    ...     experimentalist=example_experimentalist_with_parameters,
    ...     experiment_runner=example_synthetic_experiment_runner,
    ...     params={"experimentalist": {"uniform_random_sampler": {"n": 7}}}
    ... )
    >>> _ = cycle_with_parameters.run()
    >>> cycle_with_parameters.data.conditions[-1].flatten()
    array([6.33661987, 7.34916618, 6.08596494, 2.28566582, 1.9553974 ,
           5.80023149, 3.27007909])

    For the next cycle, if we wish, we can change the parameter value:
    >>> cycle_with_parameters.params["experimentalist"]["uniform_random_sampler"]\\
    ...     ["n"] = 2
    >>> _ = cycle_with_parameters.run()
    >>> cycle_with_parameters.data.conditions[-1].flatten()
    array([10.5838232 ,  9.45666031])

# Accessing "State-dependent Properties"

Some experimentalists, experiment runners and theorists require access to the values
created during the cycle execution, e.g. experimentalists which require access
to the current best theory or the observed data. These data update each cycle, and
so cannot easily be set using simple `params`.

For this case, it is possible to use "state-dependent properties" in the `params`
dictionary. These are the following strings, which will be replaced during execution by
their respective current values:

- `"%observations.ivs[-1]%"`: the last observed independent variables
- `"%observations.dvs[-1]%"`: the last observed dependent variables
- `"%observations.ivs%"`: all the observed independent variables,
concatenated into a single array
- `"%observations.dvs%"`: all the observed dependent variables,
concatenated into a single array
- `"%theories[-1]%"`: the last fitted theorist
- `"%theories%"`: all the fitted theorists

Examples:

    In the following example, we use the `"observations.ivs"` cycle property for an
    experimentalist which excludes those conditions which have
    already been seen.

    >>> metadata_1 = VariableCollection(
    ...    independent_variables=[Variable(name="x1", allowed_values=range(10))],
    ...    dependent_variables=[Variable(name="y")],
    ...    )
    >>> random_sampler_rng = np.random.default_rng(seed=180)
    >>> def custom_random_sampler(conditions, n):
    ...     sampled_conditions = random_sampler_rng.choice(conditions, size=n, replace=False)
    ...     return sampled_conditions
    >>> def exclude_conditions(conditions, excluded_conditions):
    ...     remaining_conditions = list(set(conditions) - set(excluded_conditions.flatten()))
    ...     return remaining_conditions
    >>> unobserved_data_experimentalist = make_pipeline([
    ...     metadata_1.independent_variables[0].allowed_values,
    ...     exclude_conditions,
    ...     custom_random_sampler
    ...     ]
    ... )
    >>> cycle_with_state_dep_properties = Cycle(
    ...     metadata=metadata_1,
    ...     theorist=example_theorist,
    ...     experimentalist=unobserved_data_experimentalist,
    ...     experiment_runner=example_synthetic_experiment_runner,
    ...     params={
    ...         "experimentalist": {
    ...             "exclude_conditions": {"excluded_conditions": "%observations.ivs%"},
    ...             "custom_random_sampler": {"n": 1}
    ...         }
    ...     }
    ... )

    Now we can run the cycler to generate conditions and run experiments. The first time round,
    we have the full set of 10 possible conditions to select from, and we select "2" at random:
    >>> _ = cycle_with_state_dep_properties.run()
    >>> cycle_with_state_dep_properties.data.conditions[-1]
    array([2])

    We can continue to run the cycler, each time we add more to the list of "excluded" options:
    >>> _ = cycle_with_state_dep_properties.run(num_cycles=5)
    >>> cycle_with_state_dep_properties.data.conditions
    [array([2]), array([6]), array([5]), array([7]), array([3]), array([4])]

    By using the monitor callback, we can investigate what's going on with the
    state-dependent properties:
    >>> cycle_with_state_dep_properties.monitor = lambda state: print(
    ...     np.row_stack(state.observations)[:,0]  # just the independent variable values
    ... )

    The monitor evaluates at the end of each cycle
    and shows that we've added a new observed IV each step
    >>> _ = cycle_with_state_dep_properties.run()
    [2. 6. 5. 7. 3. 4. 9.]
    >>> _ = cycle_with_state_dep_properties.run()
    [2. 6. 5. 7. 3. 4. 9. 0.]

    We deactivate the monitor by making it "None" again.
    >>> cycle_with_state_dep_properties.monitor = None

    We can continue until we've sampled all of the options:
    >>> _ = cycle_with_state_dep_properties.run(num_cycles=2)
    >>> cycle_with_state_dep_properties.data.conditions # doctest: +NORMALIZE_WHITESPACE
    [array([2]), array([6]), array([5]), array([7]), array([3]), \
    array([4]), array([9]), array([0]), array([8]), array([1])]

    If we try to evaluate it again, the experimentalist fails, as there aren't any more
    conditions which are available:
    >>> cycle_with_state_dep_properties.run()  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: a cannot be empty unless no samples are taken


# Using Alternative Executors and Planners

By switching out the `executor_collection` and/or the `planner`, we can specify a
different way of running the cycle.

## Easier Seeding with a Smarter Planner

Examples:

    In this example, we use the `Controller` which allows much more control over execution
    order. It considers the last available result and picks the matching next step. This means
    that seeding is relatively simple.
    >>> from autora.controller import Controller
    >>> def monitor(state):
    ...     print(f"MONITOR: Generated new {state.history[-1].kind}")
    >>> cycle_with_last_result_planner = Controller(
    ...     monitor=monitor,
    ...     metadata=metadata_0,
    ...     theorist=example_theorist,
    ...     experimentalist=example_experimentalist,
    ...     experiment_runner=example_synthetic_experiment_runner,
    ... )

    When we run this cycle starting with no data, we generate an experimental condition first:
    >>> _ = list(takewhile(lambda c: len(c.state.theories) < 2, cycle_with_last_result_planner))
    MONITOR: Generated new CONDITION
    MONITOR: Generated new OBSERVATION
    MONITOR: Generated new THEORY
    MONITOR: Generated new CONDITION
    MONITOR: Generated new OBSERVATION
    MONITOR: Generated new THEORY

    However, if we seed the same cycle with observations, then its first Executor will be the
    theorist:
    >>> controller_with_seed_observation = Controller(
    ...     monitor=monitor,
    ...     metadata=metadata_0,
    ...     theorist=example_theorist,
    ...     experimentalist=example_experimentalist,
    ...     experiment_runner=example_synthetic_experiment_runner,
    ... )
    >>> seed_observation = example_synthetic_experiment_runner(np.linspace(0,5,10))
    >>> controller_with_seed_observation.seed(observations=[seed_observation])

    >>> _ = next(controller_with_seed_observation)
    MONITOR: Generated new THEORY

## Arbitrary Execution Order (Toy Example)

In some cases, we need to change the order of execution of different steps completely. This might be
 useful in cases when different experimentalists or theorists are needed at different times in
 the cycle, e.g. for initial seeding, or if the _order_ of execution is the subject of the
 experiment.

Examples:

    In this example, we use a planner which suggests a different random operation at each
    step, demonstrating arbitrary execution order. We do this by modifying the planner attribute
    of an existing controller

    This might be useful in cases when different experimentalists or theorists are needed at
    different times in the cycle, e.g. for initial seeding.
    >>> from autora.controller.planner import random_operation_planner
    >>> def monitor(state):
    ...     print(f"MONITOR: Generated new {state.history[-1].kind}")
    >>> controller_with_random_planner = Controller(
    ...     planner=random_operation_planner,
    ...     monitor=monitor,
    ...     metadata=metadata_0,
    ...     theorist=example_theorist,
    ...     experimentalist=example_experimentalist,
    ...     experiment_runner=example_synthetic_experiment_runner,
    ... )

    The `random_operation_planner` depends on the python random number generator, so we seed
    it first:
    >>> from random import seed
    >>> seed(42)

    We also want to watch the logging messages from the cycle:
    >>> import logging
    >>> import sys
    >>> logging.basicConfig(format='%(levelname)s: %(message)s', stream=sys.stdout,
    ...     level=logging.INFO)

    Now we can evaluate the cycle and watch its behaviour:
    >>> def step(controller_):
    ...     try:
    ...         _ = next(controller_)
    ...     except ValueError as e:
    ...         print(f"FAILED: with {e=}")

    The first step, the theorist is selected as the random Executor, and it fails because it
    depends on there being observations to theorize against:
    >>> step(controller_with_random_planner) # i = 0
    FAILED: with e=ValueError('need at least one array to concatenate')

    The second step, a new condition is generated.
    >>> step(controller_with_random_planner) # i = 1
    MONITOR: Generated new CONDITION

    ... which is repeated on the third step as well:
    >>> step(controller_with_random_planner) # i = 2
    MONITOR: Generated new CONDITION

    On the fourth step, we generate another error when trying to run the theorist:
    >>> step(controller_with_random_planner) # i = 3
    FAILED: with e=ValueError('need at least one array to concatenate')

    On the fifth step, we generate a first real observation, so that the next time we try to run
    a theorist we are successful:
    >>> step(controller_with_random_planner) # i = 4
    MONITOR: Generated new OBSERVATION

    By the ninth iteration, there are observations which the theorist can use, and it succeeds.
    >>> _ = list(takewhile(lambda c: len(c.state.theories) < 1, controller_with_random_planner))
    MONITOR: Generated new CONDITION
    MONITOR: Generated new CONDITION
    MONITOR: Generated new CONDITION
    MONITOR: Generated new THEORY

## Arbitrary Executors and Planners

In some cases, we need to go beyond adding different orders of planning the three
`experimentalist`, `experiment_runner` and `theorist` and build more complex cycles with
different Executors for different states.

For instance, there might be a situation where at the start, the main "active" experimentalist
can't be run as it needs one or more theories as input.
Once there are at least two theories, then the active experimentalist _can_ be run.
One method to handle this is to run a "seed" experimentalist until the main experimentalist can
be used.

In these cases, we need full control over (and have full responsibility for) the planners and
executors.

Examples:
    The theory we'll try to discover is:
    >>> def ground_truth(x, m=3.5, c=1):
    ...     return m * x + c
    >>> rng = np.random.default_rng(seed=180)
    >>> def experiment_runner(x):
    ...     return ground_truth(x) + rng.normal(0, 0.1)
    >>> metadata_2 = VariableCollection(
    ...    independent_variables=[Variable(name="x1", value_range=(-10, 10))],
    ...    dependent_variables=[Variable(name="y", value_range=(-100, 100))],
    ...    )

    We now define a planner which chooses a different experimentalist when supplied with no data
    versus some data.
    >>> from autora.controller.protocol import ResultKind
    >>> from autora.controller.planner import last_result_kind_planner
    >>> def seeding_planner(state):
    ...     # We're going to reuse the "last_available_result" planner, and modify its output.
    ...     next_function = last_result_kind_planner(state)
    ...     if next_function == "experimentalist":
    ...         if len(state.theories) >= 2:
    ...             return "main_experimentalist"
    ...         else:
    ...             return "seed_experimentalist"
    ...     else:
    ...         return next_function

    Now we can see what would happen with a particular state. If there are no results,
    then we get the seed experimentalist:
    >>> from autora.controller.state import History
    >>> seeding_planner(History())
    'seed_experimentalist'

    ... and we also get the seed experimentalist if the last result was a theory and there are less
    than two theories:
    >>> seeding_planner(History(theories=['a single theory']))
    'seed_experimentalist'

    ... whereas if we have at least two theories to work on, we get the main experimentalist:
    >>> seeding_planner(History(theories=['a theory', 'another theory']))
    'main_experimentalist'

    If we had a condition last, we choose the experiment runner next:
    >>> seeding_planner(History(conditions=['a condition']))
    'experiment_runner'

    If we had an observation last, we choose the theorist next:
    >>> seeding_planner(History(observations=['an observation']))
    'theorist'

    Now we need to define an executor collection to handle the actual execution steps.
    >>> from autora.experimentalist.pipeline import make_pipeline, Pipeline
    >>> from autora.experimentalist.sampler.random import random_sampler
    >>> from functools import partial

    Wen can run the seed pipeline with no data:
    >>> experimentalist_which_needs_no_data = make_pipeline([
    ...     np.linspace(*metadata_2.independent_variables[0].value_range, 1_000),
    ...     partial(random_sampler, n=10)]
    ... )
    >>> np.array(experimentalist_which_needs_no_data())
    array([ 6.71671672, -0.73073073, -5.05505506,  6.13613614,  0.03003003,
            4.59459459,  2.79279279,  5.43543544, -1.65165165,  8.0980981 ])


    ... whereas we need some model for this sampler:
    >>> from autora.experimentalist.sampler.model_disagreement import model_disagreement_sampler
    >>> experimentalist_which_needs_a_theory = Pipeline([
    ...     ('pool', np.linspace(*metadata_2.independent_variables[0].value_range, 1_000)),
    ...     ('sampler', partial(model_disagreement_sampler, num_samples=5)),])
    >>> experimentalist_which_needs_a_theory()
    Traceback (most recent call last):
    ...
    TypeError: model_disagreement_sampler() missing 1 required positional argument: 'models'

    We'll have to provide the models during the cycle run.

    We need a reasonable theorist for this situation. For this problem, a linear regressor will
    suffice.
    >>> t = LinearRegression()

    Let's test the theorist for the ideal case – lots of data:
    >>> X = np.linspace(*metadata_2.independent_variables[0].value_range, 1_000).reshape(-1, 1)
    >>> tfitted = t.fit(X, experiment_runner(X))
    >>> f"m = {tfitted.coef_[0][0]:.2f}, c = {tfitted.intercept_[0]:.2f}"
    'm = 3.50, c = 1.04'

    This seems to work fine.

    Now we can define the executor component. We'll use a factory method to generate the
    collection:
    >>> from autora.controller.executor import make_online_executor_collection
    >>> executor_collection = make_online_executor_collection([
    ...     ("seed_experimentalist", "experimentalist", experimentalist_which_needs_no_data),
    ...     ("main_experimentalist", "experimentalist", experimentalist_which_needs_a_theory),
    ...     ("theorist", "theorist", LinearRegression()),
    ...     ("experiment_runner", "experiment_runner", experiment_runner),
    ... ])

    We need some special parameters to handle the main experimentalist, so we specify those:
    >>> params = {"experimentalist": {"sampler": {"models": "%theories%"}}}

    Warning: the dictionary `{"sampler": {"models": "%theories%"}}` above is shared by
    both the seed and main experimentalists. This behavior may change in future to allow separate
    parameter dictionaries for each executor in the collection.

    We now instantiate the controller:
    >>> from autora.controller.base import BaseController
    >>> from autora.controller.state import History
    >>> c = BaseController(
    ...         state=History(metadata=metadata_2, params=params),
    ...         planner=seeding_planner,
    ...         executor_collection=executor_collection
    ... )
    >>> c  # doctest: +ELLIPSIS
    <...BaseController object at 0x...>

    >>> class PrintHandler(logging.Handler):
    ...     def emit(self, record):
    ...         print(self.format(record))

    On the first step, we generate a condition sampled randomly across the whole domain (as we
    expected):
    >>> next(c).state.history[-1]  # doctest: +NORMALIZE_WHITESPACE
    Result(data=array([ 9.4994995 , -8.17817818, -1.19119119,  8.6986987 ,  7.45745746,
                      -6.93693694,  8.05805806, -1.45145145, -5.97597598,  1.57157157]),
           kind=ResultKind.CONDITION)

    After three more steps, we generate a new condition, which again is sampled across the whole
    domain. Here we iterate the controller until we've got two sets of conditions:
    >>> _ = list(takewhile(lambda c: len(c.state.conditions) < 2, c))
    >>> c.state.history[-1]  # doctest: +NORMALIZE_WHITESPACE
    Result(data=array([ 1.57157157, -3.93393393, -0.47047047, -4.47447447,  8.43843844,
                        6.17617618, -3.49349349, -8.998999  ,  4.93493493,  2.25225225]),
           kind=ResultKind.CONDITION)

    Once we have two theories:
    >>> _ = list(takewhile(lambda c: len(c.state.theories) < 2, c))
    >>> c.state.theories
    [LinearRegression(), LinearRegression()]

    ... when we run the next step, we'll get the main experimentalist, which samples five points
    from the extreme parts of the problem domain where the disagreement between the two theories
    is the greatest:
    >>> next(c).state.history[-1]  # doctest: +NORMALIZE_WHITESPACE
    Result(data=array([-10.       ,  -9.97997998,  -9.95995996,  -9.93993994,  -9.91991992]),
           kind=ResultKind.CONDITION)

"""
from .controller import Controller
from .cycle import Cycle
