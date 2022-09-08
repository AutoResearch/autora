import numpy as np
from sklearn.model_selection import train_test_split

from aer_bms.skl.bms import BMS


def generate_noisy_constant_data(
    const: float = 0.5, epsilon: float = 0.01, num: int = 1000, seed: int = 42
):
    X = np.expand_dims(np.linspace(start=0, stop=1, num=num), 1)
    y = np.random.default_rng(seed).normal(loc=const, scale=epsilon, size=num)
    return X, y, const, epsilon


def test_constant_model():
    X, y, const, epsilon = generate_noisy_constant_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimator = BMS()
    assert estimator is not None
    estimator.fit(X_train, y_train)
    print(estimator.model_)


if __name__ == "__main__":
    test_constant_model()
