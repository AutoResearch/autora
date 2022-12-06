import pytest
from skl.darts import DARTSClassifier
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split


@pytest.fixture
def classification_data():
    x, y = make_classification(random_state=180)


def test_darts_classifier(classification_data):
    x, y = classification_data

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=181)
    for classifier in [GaussianProcessClassifier(), DARTSClassifier()]:
        classifier.fit(x_train, y_train)
        score = classifier.score(x_test, y_test)
        print(f"\n{classifier=} {score=}")
