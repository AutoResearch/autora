"""
Provides an interface for loading and saving synthetic experiments.

Examples:
    The registry is accessed using the `retrieve` function:
    >>> from autora.synthetic import retrieve
    >>> s = retrieve("weber_fechner")  # the Weber-Fechner Law experiment

    The synthetic experiement `s` has properties like the name of the experiment:
    >>> s.name
    'Weber-Fechner Law'

    ... a callable which can be used to generate a valid metadata description:
    >>> s.metadata_callable  # doctest: +ELLIPSIS
    <function weber_fechner_metadata at 0x...>
    >>> s.metadata_callable()  # doctest: +ELLIPSIS
    VariableCollection(...)

    ... the experiment runner which can be called to generate experimental results:
    >>> import numpy as np
    >>> s.experiment_runner(np.array([(0.1, 0.2)]))  # doctest: +ELLIPSIS
    array([[0...]])

    ... a function to generate the full domain of the data (if possible)
    >>> x, y = s.data_callable(s.metadata_callable())
    >>> x   # doctest: +ELLIPSIS
    array([[0.01      , 0.01      ],...

    ... a function to plot the ground truth:
    >>> s.plotter()

    ... against a fitted model if it exists:
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression().fit(x, y)
    >>> s.plotter(model)

    These can be used to run a full experimental cycle
    >>> from autora.experimentalist.pipeline import make_pipeline
    >>> from autora.experimentalist.pooler.general_pool import grid_pool
    >>> from autora.experimentalist.sampler.random import random_sampler
    >>> from functools import partial
    >>> metadata = s.metadata_callable()
    >>> pool = partial(grid_pool, ivs=metadata.independent_variables)
    >>> sampler = partial(random_sampler, n=20)
    >>> experimentalist_pipeline = make_pipeline([pool, sampler])

    >>> from autora.cycle import Cycle
    >>> theorist = LinearRegression()

    >>> cycle = Cycle(metadata=metadata, experimentalist=experimentalist_pipeline,
    ...               experiment_runner=s.experiment_runner, theorist=theorist)

    >>> c = cycle.run(10)
    >>> c.data.theories[-1].coef_   # doctest: +ELLIPSIS
    array([-0...,  0...])
"""

from autora.synthetic import data
from autora.synthetic._inventory import register, retrieve
