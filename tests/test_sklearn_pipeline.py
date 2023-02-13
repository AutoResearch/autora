import numpy as np
import sklearn.pipeline as skp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from autora.cycle import Cycle
from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.sampler import random_sampler
from autora.variable import Variable, VariableCollection


def test_skpipe_theory_copy():
    """Checks that a deep copy is performed when the autora.Cycle is copying the theorist."""

    X = np.linspace(0, 1, 10)

    # Variable Metadata
    study_metadata = VariableCollection(
        independent_variables=[
            Variable(name="x", units="cm", allowed_values=np.linspace(0, 1, 100)),
        ],
        dependent_variables=[Variable(name="class", allowed_values=[0, 1])],
    )

    # Theorist with skl Pipeline
    clf = skp.Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])

    # Experimentalist
    experimentalist = Pipeline(
        [
            ("pool", X),
            ("sampler", random_sampler),
        ],
        params={
            "sampler": {"n": 15},
        },
    )

    # Experiment Runner
    def experiment_runner(xs):
        y_return = np.where(xs > 0.5, 1, 0)
        return y_return

    cycle = Cycle(
        metadata=study_metadata,
        theorist=clf,
        experimentalist=experimentalist,
        experiment_runner=experiment_runner,
    )
    cycle.run(6)

    # Check that the memory space of all estimators are different
    l_memory = [id(s["lr"]) for s in cycle.data.theories]
    n_unique = len(np.unique(l_memory))
    n_theories = len(cycle.data.theories)
    assert n_unique == n_theories
