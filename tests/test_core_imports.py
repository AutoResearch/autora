def test_core_imports():
    # Only include default packages installed by `pip install autora` without any extras here
    import autora  # noqa
    import autora.experiment_runner.synthetic  # noqa
    import autora.experimentalist  # noqa
    import autora.utils  # noqa
    import autora.variable  # noqa
    import autora.workflow  # noqa
