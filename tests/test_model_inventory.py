import pytest  # noqa: 401

import autora.synthetic
from autora.variable import VariableCollection


def test_model_registration():
    # We can register a model and retrieve it
    autora.synthetic.register(id="empty")
    empty = autora.synthetic.retrieve("empty")
    assert empty.id == "empty"
    assert empty.name is None

    # We can register another model and retrieve it as well
    autora.synthetic.register(
        id="only_metadata", metadata_callable=lambda: VariableCollection()
    )
    only_metadata = autora.synthetic.retrieve("only_metadata")
    assert only_metadata.id == "only_metadata"
    assert only_metadata.metadata_callable()

    # We can still retrieve the first model and it is equal to the first version
    empty_copy = autora.synthetic.retrieve("empty")
    assert empty_copy == empty


def test_model_retrieval():
    for id in ["weber_fechner", "expected_value", "prospect_theory"]:
        model = autora.synthetic.retrieve(id)
        assert model.id == id
