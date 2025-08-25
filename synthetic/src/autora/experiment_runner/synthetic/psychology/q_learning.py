from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def _check_in_0_1_range(x, name):
    if not (0 <= x <= 1):
        raise ValueError(
            f"Value of {name} must be in [0, 1] range. Found value of {x}."
        )


class AgentQ:
    """An agent that runs simple Q-learning for an n-armed bandits tasks.

    Attributes:
      alpha: The agent's learning rate
      beta: The agent's softmax temperature
      q: The agent's current estimate of the reward probability on each arm
    """

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 3.0,
        n_actions: int = 2,
        forget_rate: float = 0.0,
        perseverance_bias: float = 0.0,
        correlated_reward: bool = False,
    ):
        """Update the agent after one step of the task.

        Args:
          alpha: scalar learning rate
          beta: scalar softmax inverse temperature parameter.
          n_actions: number of actions (default=2)
          forgetting_rate: rate at which q values decay toward the initial values (default=0)
          perseveration_bias: rate at which q values move toward previous action (default=0)
        """
        self._prev_choice = -1
        self._alpha = alpha
        self._beta = beta
        self._n_actions = n_actions
        self._forget_rate = forget_rate
        self._perseverance_bias = perseverance_bias
        self._correlated_reward = correlated_reward
        self._q_init = 0.5
        self.new_sess()

        _check_in_0_1_range(alpha, "alpha")
        _check_in_0_1_range(forget_rate, "forget_rate")

    def new_sess(self):
        """Reset the agent for the beginning of a new session."""
        self._q = self._q_init * np.ones(self._n_actions)
        self._prev_choice = -1

    def get_choice_probs(self) -> np.ndarray:
        """Compute the choice probabilities as softmax over q."""
        decision_variable = np.exp(self.q * self._beta)
        choice_probs = decision_variable / np.sum(decision_variable)
        return choice_probs

    def get_choice(self) -> int:
        """Sample a choice, given the agent's current internal state."""
        choice_probs = self.get_choice_probs()
        choice = np.random.choice(self._n_actions, p=choice_probs)
        return choice

    def update(self, choice: int, reward: float):
        """Update the agent after one step of the task.

        Args:
          choice: The choice made by the agent. 0 or 1
          reward: The reward received by the agent. 0 or 1
        """

        # Forgetting - restore q-values of non-chosen actions towards the initial value
        non_chosen_action = np.arange(self._n_actions) != choice
        self._q[non_chosen_action] = (1 - self._forget_rate) * self._q[
            non_chosen_action
        ] + self._forget_rate * self._q_init

        # Reward-based update - Update chosen q for chosen action with observed reward
        q_reward_update = -self._alpha * self._q[choice] + self._alpha * reward

        # Correlated update - Update non-chosen q for non-chosen action with observed reward
        if self._correlated_reward:
            # index_correlated_update = self._n_actions - choice - 1
            # self._q[index_correlated_update] =
            # (1 - self._alpha) * self._q[index_correlated_update] + self._alpha * (1 - reward)
            # alternative implementation - not dependent on reward but on reward-based update
            index_correlated_update = self._n_actions - 1 - choice
            self._q[index_correlated_update] -= 0.5 * q_reward_update

        # Memorize current choice for perseveration
        self._prev_choice = choice

        self._q[choice] += q_reward_update

    @property
    def q(self):
        q = self._q.copy()
        if self._prev_choice != -1:
            q[self._prev_choice] += self._perseverance_bias
        return q


def q_learning(
    name="Q-Learning",
    learning_rate: float = 0.2,
    decision_noise: float = 3.0,
    n_actions: int = 2,
    forget_rate: float = 0.0,
    perseverance_bias: float = 0.0,
    correlated_reward: bool = False,
):
    """
    An agent that runs simple Q-learning for an n-armed bandits tasks.

    Args:
        name: name of the experiment
        trials: number of trials
        learning_rate: learning rate for Q-learning
        decision_noise: softmax parameter for decision noise
        n_actions: number of actions
        forget_rate: rate of forgetting
        perseverance_bias: bias towards choosing the previously chosen action
        correlated_reward: whether rewards are correlated

    Examples:
        >>> experiment = q_learning()

        # The runner can accept numpy arrays or pandas DataFrames, but the return value will
        # always be a list of numpy arrays. Each array corresponds to the choices made by the agent
        # for each trial in the input. Thus, arrays have shape (n_trials, n_actions).
        >>> experiment.run(np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]]),
        ...                random_state=42)
        [array([[1., 0.],
               [0., 1.],
               [0., 1.],
               [0., 1.],
               [1., 0.],
               [1., 0.]])]

        # The runner can accept pandas DataFrames. Each cell of the DataFrame should contain a
        # numpy array with shape (n_trials, n_actions). The return value will be a list of numpy
        # arrays, each corresponding to the choices made by the agent for each trial in the input.
        >>> experiment.run(
        ...     pd.DataFrame(
        ...         {'reward array': [np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])]}),
        ...     random_state = 42)
        [array([[1., 0.],
               [0., 1.],
               [0., 1.],
               [0., 1.],
               [1., 0.],
               [1., 0.]])]
    """

    params = dict(
        name=name,
        trials=100,
        learning_rate=learning_rate,
        decision_noise=decision_noise,
        n_actions=n_actions,
        forget_rate=forget_rate,
        perseverance_bias=perseverance_bias,
        correlated_reward=correlated_reward,
    )

    iv1 = IV(
        name="reward array",
        units="reward",
        variable_label="Reward Sequence",
        type=ValueType.BOOLEAN,
    )

    dv1 = DV(
        name="choice array",
        units="actions",
        variable_label="Action Sequence",
        type=ValueType.REAL,
    )

    variables = VariableCollection(
        independent_variables=[iv1],
        dependent_variables=[dv1],
    )

    def run_AgentQ(rewards):
        if rewards.shape[1] != n_actions:
            Warning(
                "Number of actions in rewards does not match n_actions. Will use "
                + str(rewards.shape[1] + " actions.")
            )
        num_trials = rewards.shape[0]

        y = np.zeros(rewards.shape)
        choice_proba = np.zeros(rewards.shape)

        agent = AgentQ(
            alpha=learning_rate,
            beta=decision_noise,
            n_actions=rewards.shape[1],
            forget_rate=forget_rate,
            perseverance_bias=perseverance_bias,
            correlated_reward=correlated_reward,
        )

        for i in range(num_trials):
            proba = agent.get_choice_probs()
            choice = agent.get_choice()
            y[i, choice] = 1
            choice_proba[i] = proba
            reward = rewards[i, choice]
            agent.update(choice, reward)
        return y, choice_proba

    def run(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        random_state: Optional[int] = None,
        return_choice_probabilities=False,
    ):
        if random_state is not None:
            np.random.seed(random_state)

        Y = list()
        Y_proba = list()
        if isinstance(conditions, pd.DataFrame):
            for index, session in conditions.iterrows():
                rewards = session[0]
                choice, choice_proba = run_AgentQ(rewards)
                Y.append(choice)
                Y_proba.append(choice_proba)
        elif isinstance(conditions, np.ndarray):
            choice, choice_proba = run_AgentQ(conditions)
            Y.append(choice)
            Y_proba.append(choice_proba)

        if return_choice_probabilities:
            return Y, Y_proba
        else:
            return Y

    ground_truth = partial(run)

    def domain():
        return None

    def plotter():
        raise NotImplementedError

    collection = SyntheticExperimentCollection(
        name=name,
        description=q_learning.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=q_learning,
    )
    return collection
