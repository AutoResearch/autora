import pytest
import torch
from skl.darts import DARTSClassifier
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.double)


@pytest.fixture
def classification_data():
    x, y = make_classification(random_state=180)
    return x, y


def test_darts_classifier(classification_data):
    x, y = classification_data

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=181)
    for classifier in [GaussianProcessClassifier(), DARTSClassifier()]:
        classifier.fit(x_train, y_train)

        predictions = classifier.predict(x_test)
        assert predictions is not None

        prediction_probabilities = classifier.predict_proba(x_test)
        assert prediction_probabilities is not None

        score = classifier.score(x_test, y_test)
        print(f"\n{classifier=} {score=}")
