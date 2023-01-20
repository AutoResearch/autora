import warnings

import pytest  # noqa: 401

import autora.synthetic

warnings.filterwarnings("ignore")


def test_model_recovery():
    for id in ["weber_fechner", "expected_value", "prospect_theory"]:
        model = autora.synthetic.retrieve(id)
        print(model)
