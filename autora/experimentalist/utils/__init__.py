from __future__ import annotations

import collections
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np

from autora.experimentalist.pipeline import Pipeline, _ExperimentalSequence, _StepType


class ArrayPipelineWrapper(Pipeline):
    """
    A pipeline which uses arrays as its internal representation of data, and accepts/ouputs
    sequences.

    This is useful when Pipes which expect finite arrays as input and produce arrays as output,
    rather than a (potentially unbounded) iterable of experimental conditions.

    This type of pipeline only accepts bounded Sequences as inputs.

    Examples:
        The empty ArrayPipelineWrapper just returns the input sequence as arrays
        >>> p0 = ArrayPipelineWrapper([])
        >>> list(p0.run(zip(range(5), range(5,15))))
        [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

        The intermediate pipe steps can use Numpy functionality by default:
        >>> p1 = ArrayPipelineWrapper([
        ...     ("mask", lambda cs: cs[[True, False, True, True, False]])
        ... ])
        >>> list(p1.run(range(5)))
        [(0,), (2,), (3,)]

        >>> def echo(x):
        ...     print(f"within the pipeline, x is a numpy.array: {x=}")
        ...     return x
        >>> p2 = ArrayPipelineWrapper([
        ...     ("mask", lambda x: x[[True, False, True, True, False]]),
        ...     ("echo", echo)
        ... ], array_type="numpy.array")
        >>> p2_output = p2.run(range(5))  # doctest: +NORMALIZE_WHITESPACE
        within the pipeline, x is a numpy.array: x=array([[0], [2], [3]])
        >>> list(p2_output)
        [array([0]), array([2]), array([3])]

        You can also use this with a pooler which produces arrays:
        >>> p3 = ArrayPipelineWrapper([
        ...     ("pool", np.arange(10).reshape(-1, 2)),
        ...     ("mask", lambda cs: cs[[True, False, True, True, False]])
        ... ])
        >>> list(p3.run())
        [array([0, 1]), array([4, 5]), array([6, 7])]

    """

    def __init__(
        self,
        steps: Optional[Sequence[_StepType]] = None,
        params: Optional[Dict[str, Any]] = None,
        array_type: Literal["numpy.array", "numpy.rec.array"] = "numpy.rec.array",
    ):
        steps_with_wrappers: List[_StepType] = []

        if (
            steps is None
            or len(steps) == 0
            or not isinstance(steps[0][1], (np.ndarray, np.recarray))
        ):
            steps_with_wrappers.extend(
                [("sequence_to_array", partial(sequence_to_array, type=array_type))]
            )
        else:
            pass  # we don't need to convert the input – we just assume the first step is a pool
            # which produces a valid array

        if steps is None:
            pass
        else:
            steps_with_wrappers.extend(steps)

        steps_with_wrappers.extend([("array_to_sequence", partial(array_to_sequence))])

        super().__init__(steps=steps_with_wrappers, params=params)
        return


def sequence_to_array(
    input: _ExperimentalSequence, type: Literal["numpy.array", "numpy.rec.array"]
):
    """
    Converts a finite sequence of experimental conditions into a 2D array.

    See also: [array_to_sequence][autora.experimentalist.pipeline.array_to_sequence]

    Examples:

        A simple range object can be converted into an array of dimension 2:
        >>> sequence_to_array(range(5), type="numpy.array") # doctest: +NORMALIZE_WHITESPACE
        array([[0], [1], [2], [3], [4]])

        An alternative representation is the record array, also of dimension 2:
        >>> sequence_to_array(range(5), type="numpy.rec.array") # doctest: +NORMALIZE_WHITESPACE
        rec.array([(0,), (1,), (2,), (3,), (4,)], dtype=[('f0', '<i8')])

        For mixed datatypes, it is sensible to use a record array:
        >>> sequence_to_array(zip(range(5), "abcde"), type="numpy.rec.array"
        ...     )  # doctest: +NORMALIZE_WHITESPACE
        rec.array([(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
            dtype=[('f0', '<i8'), ('f1', '<U1')])

        ... otherwise, the highest-level type common to all the inputs will be used:
        >>> sequence_to_array(zip(range(5), "abcde"), type="numpy.array"
        ...     )  # doctest: +NORMALIZE_WHITESPACE
        array([['0', 'a'], ['1', 'b'], ['2', 'c'], ['3', 'd'], ['4', 'e']],  dtype='<U21')

        Single strings are broken into characters:
        >>> sequence_to_array("abcde", type="numpy.rec.array")  # doctest: +NORMALIZE_WHITESPACE
        rec.array([('a',), ('b',), ('c',), ('d',), ('e',)], dtype=[('f0', '<U1')])

        >>> sequence_to_array("abcde", type="numpy.array")  # doctest: +NORMALIZE_WHITESPACE
        array([['a'], ['b'], ['c'], ['d'], ['e']], dtype='<U1')

        Multiple strings are treated as individual entries:
        >>> sequence_to_array(["abc", "de"], type="numpy.rec.array"
        ... )  # doctest: +NORMALIZE_WHITESPACE
        rec.array([('abc',), ('de',)], dtype=[('f0', '<U3')])

        >>> sequence_to_array(["abc", "de"], type="numpy.array")  # doctest: +NORMALIZE_WHITESPACE
        array([['abc'], ['de']], dtype='<U3')

    """
    deque = collections.deque(input)

    if type == "numpy.array":
        return np.array(deque).reshape((len(deque), -1))
    if type == "numpy.rec.array":
        if isinstance(deque[0], (str, int, float, complex)):
            return np.core.records.fromrecords([(d,) for d in deque])
        else:
            return np.core.records.fromrecords(deque)
    else:
        raise NotImplementedError(f"{type=} not implemented")


def array_to_sequence(input: Union[np.array, np.recarray]):
    """
    Convert an array of experimental conditions into an iterable of smaller arrays.

    See also: [sequence_to_array][autora.experimentalist.pipeline.sequence_to_array]

    Examples:

        We start with an array:
        >>> a0 = np.arange(10).reshape(-1,2)
        >>> a0
        array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]])

        The sequence is created as a generator object
        >>> array_to_sequence(a0)  # doctest: +ELLIPSIS
        <generator object array_to_sequence at 0x...>

        To see the sequence, we can convert it into a list:
        >>> l0 = list(array_to_sequence(a0))
        >>> l0
        [array([0, 1]), array([2, 3]), array([4, 5]), array([6, 7]), array([8, 9])]

        The individual rows are themselves 1-dimensional arrays:
        >>> l0[0]
        array([0, 1])

        The rows can be subscripted as usual:
        >>> l0[2][1]
        5

        We can also use a record array:
        >>> a1 = np.rec.fromarrays([range(5), list("abcde")])
        >>> a1
        rec.array([(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
                  dtype=[('f0', '<i8'), ('f1', '<U1')])

        This is converted into records:
        >>> l1 = list(array_to_sequence(a1))
        >>> l1
        [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]

        The elements of the list are numpy.records
        >>> type(l1[0])
        <class 'numpy.record'>

    """
    if isinstance(input, (np.ndarray, np.recarray)):
        for a in input:
            yield a
    else:
        raise NotImplementedError(f"{type(input)} not supported")
