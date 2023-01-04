import numpy as np
import pytest

from autora.skl.bms import BMSRegressor
from autora.theorist.bms import Tree


@pytest.fixture
def curve_to_fit():
    x = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = (x**3.0) + (2.0 * x**2.0) + (17.0 * x) - 1
    return x, y


def test_bms_models(curve_to_fit):
    x, y = curve_to_fit
    regressor = BMSRegressor(epochs=100)

    regressor.fit(x, y)

    print(regressor.models_)

    assert len(regressor.models_) == len(regressor.ts)  # Currently hardcoded
    for model in regressor.models_:
        assert isinstance(model, Tree)
