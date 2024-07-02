from __future__ import annotations

import collections

import numpy as np
import numpy.typing


def sequence_to_array(iterable):
    """
    Converts a finite sequence of experimental conditions into a 2D numpy.array.

    See also: [array_to_sequence][autora.experimentalist.utils.array_to_sequence]

    Examples:

        A simple range object can be converted into an array of dimension 2:
        >>> sequence_to_array(range(5)) # doctest: +NORMALIZE_WHITESPACE
        array([[0], [1], [2], [3], [4]])

        For mixed datatypes, the highest-level type common to all the inputs will be used, so
        consider using [_sequence_to_recarray][autora.experimentalist.utils._sequence_to_recarray]
        instead.
        >>> sequence_to_array(zip(range(5), "abcde"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        array([['0', 'a'], ['1', 'b'], ['2', 'c'], ['3', 'd'], ['4', 'e']],  dtype='<U...')

        Single strings are broken into characters:
        >>> sequence_to_array("abcde")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        array([['a'], ['b'], ['c'], ['d'], ['e']], dtype='<U...')

        Multiple strings are treated as individual entries:
        >>> sequence_to_array(["abc", "de"])  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        array([['abc'], ['de']], dtype='<U...')

    """
    deque = collections.deque(iterable)
    array = np.array(deque).reshape((len(deque), -1))
    return array


def sequence_to_recarray(iterable):
    """
    Converts a finite sequence of experimental conditions into a numpy recarray.

    See also: [array_to_sequence][autora.experimentalist.utils.array_to_sequence]

    Examples:

        A simple range object is converted into a recarray of dimension 2:
        >>> sequence_to_recarray(range(5)) # doctest: +NORMALIZE_WHITESPACE
        rec.array([(0,), (1,), (2,), (3,), (4,)], dtype=[('f0', '<i...')])

        Mixed datatypes lead to multiple output types:
        >>> sequence_to_recarray(zip(range(5), "abcde"))  # doctest: +NORMALIZE_WHITESPACE
        rec.array([(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
            dtype=[('f0', '<i...'), ('f1', '<U1')])

        Single strings are broken into characters:
        >>> sequence_to_recarray("abcde")  # doctest: +NORMALIZE_WHITESPACE
        rec.array([('a',), ('b',), ('c',), ('d',), ('e',)], dtype=[('f0', '<U1')])

        Multiple strings are treated as individual entries:
        >>> sequence_to_recarray(["abc", "de"])  # doctest: +NORMALIZE_WHITESPACE
        rec.array([('abc',), ('de',)], dtype=[('f0', '<U3')])

    """
    deque = collections.deque(iterable)

    if isinstance(deque[0], (str, int, float, complex)):
        recarray = np.core.records.fromrecords([(d,) for d in deque])
    else:
        recarray = np.core.records.fromrecords(deque)

    return recarray


def array_to_sequence(input: numpy.typing.ArrayLike):
    """
    Convert an array of experimental conditions into an iterable of smaller arrays.

    See also:
        - [sequence_to_array][autora.experimentalist.utils.sequence_to_array]
        - [sequence_to_array][autora.experimentalist.utils.sequence_to_recarray]

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
                  dtype=[('f0', '<i...'), ('f1', '<U1')])

        This is converted into records:
        >>> l1 = list(array_to_sequence(a1))
        >>> l1
        [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]

        The elements of the list are numpy.records
        >>> type(l1[0])
        <class 'numpy.record'>

    """
    assert isinstance(input, (np.ndarray, np.recarray))

    for a in input:
        yield a
