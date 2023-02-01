import autora.synthetic
from autora.variable import VariableCollection


def test_model_registration():
    # We can register a model and retrieve it
    autora.synthetic.register("empty")
    empty = autora.synthetic.retrieve("empty")
    assert empty.id_ == "empty"
    assert empty.name is None

    # We can register another model and retrieve it as well
    autora.synthetic.register(id_="only_metadata", metadata=VariableCollection())
    only_metadata = autora.synthetic.retrieve("only_metadata")
    assert only_metadata.id_ == "only_metadata"
    assert only_metadata.metadata is not None

    # We can still retrieve the first model, and it is equal to the first version
    empty_copy = autora.synthetic.retrieve("empty")
    assert empty_copy == empty
