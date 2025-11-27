"""Classes to represent cycle state $S$ as $S_n = S_{0} + \\sum_{i=1}^n \\Delta S_{i}$."""

from __future__ import annotations

import copy
import inspect
import logging
import warnings
from collections import UserDict
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from functools import singledispatch, wraps
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from autora.variable import VariableCollection

_logger = logging.getLogger(__name__)

T = TypeVar("T")
C = TypeVar("C", covariant=True)


class DeltaAddable(Protocol[C]):
    """A class which a Delta or other Mapping can be added to, returning the same class"""

    def __add__(self: C, other: Union[Delta, Mapping]) -> C:
        ...


S = TypeVar("S", bound=DeltaAddable)


@dataclass(frozen=True)
class State:
    """
    Base object for dataclasses which use the Delta mechanism.

    Examples:
        >>> from dataclasses import dataclass, field
        >>> from typing import List, Optional

        We define a dataclass where each field (which is going to be delta-ed) has additional
        metadata "delta" which describes its delta behaviour.
        >>> @dataclass(frozen=True)
        ... class ListState(State):
        ...    l: List = field(default_factory=list, metadata={"delta": "extend"})
        ...    m: List = field(default_factory=list, metadata={"delta": "replace"})

        Now we instantiate the dataclass...
        >>> l = ListState(l=list("abc"), m=list("xyz"))
        >>> l
        ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... and can add deltas to it. `l` will be extended:
        >>> l + Delta(l=list("def"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['x', 'y', 'z'])

        ... wheras `m` will be replaced:
        >>> l + Delta(m=list("uvw"))
        ListState(l=['a', 'b', 'c'], m=['u', 'v', 'w'])

        ... they can be chained:
        >>> l + Delta(l=list("def")) + Delta(m=list("uvw"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['u', 'v', 'w'])

        ... and we update multiple fields with one Delta:
        >>> l + Delta(l=list("ghi"), m=list("rst"))
        ListState(l=['a', 'b', 'c', 'g', 'h', 'i'], m=['r', 's', 't'])

        A non-existent field will be ignored:
        >>> l + Delta(o="not a field")
        ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... but will trigger a warning:
        >>> with warnings.catch_warnings(record=True) as w:
        ...     _ = l + Delta(o="not a field")
        ...     print(w[0].message) # doctest: +NORMALIZE_WHITESPACE
        These fields: ['o'] could not be used to update ListState,
        which has these fields & aliases: ['l', 'm']

        We can also use the `.update` method to do the same thing:
        >>> l.update(l=list("ghi"), m=list("rst"))
        ListState(l=['a', 'b', 'c', 'g', 'h', 'i'], m=['r', 's', 't'])

        We can also define fields which `append` the last result:
        >>> @dataclass(frozen=True)
        ... class AppendState(State):
        ...    n: List = field(default_factory=list, metadata={"delta": "append"})

        >>> m = AppendState(n=list("ɑβɣ"))
        >>> m
        AppendState(n=['ɑ', 'β', 'ɣ'])

        `n` will be appended:
        >>> m + Delta(n="∂")
        AppendState(n=['ɑ', 'β', 'ɣ', '∂'])

        The metadata key "converter" is used to coerce types (inspired by
        [PEP 712](https://peps.python.org/pep-0712/)):
        >>> @dataclass(frozen=True)
        ... class CoerceStateList(State):
        ...    o: Optional[List] = field(default=None, metadata={"delta": "replace"})
        ...    p: List = field(default_factory=list, metadata={"delta": "replace",
        ...                                                    "converter": list})

        >>> r = CoerceStateList()

        If there is no `metadata["converter"]` set for a field, no coercion occurs
        >>> r + Delta(o="not a list")
        CoerceStateList(o='not a list', p=[])

        If there is a `metadata["converter"]` set for a field, the data are coerced:
        >>> r + Delta(p="not a list")
        CoerceStateList(o=None, p=['n', 'o', 't', ' ', 'a', ' ', 'l', 'i', 's', 't'])

        If the input data are of the correct type, they are returned unaltered:
        >>> r + Delta(p=["a", "list"])
        CoerceStateList(o=None, p=['a', 'list'])

        With a converter, inputs are converted to the type output by the converter:
        >>> @dataclass(frozen=True)
        ... class CoerceStateDataFrame(State):
        ...    q: pd.DataFrame = field(default_factory=pd.DataFrame,
        ...                            metadata={"delta": "replace",
        ...                                      "converter": pd.DataFrame})

        If the type is already correct, the object is passed to the converter,
        but should be returned unchanged:
        >>> s = CoerceStateDataFrame()
        >>> (s + Delta(q=pd.DataFrame([("a",1,"alpha"), ("b",2,"beta")], columns=list("xyz")))).q
           x  y      z
        0  a  1  alpha
        1  b  2   beta

        If the type is not correct, the object is converted if possible. For a dataframe,
        we can convert records:
        >>> (s + Delta(q=[("a",1,"alpha"), ("b",2,"beta")])).q
           0  1      2
        0  a  1  alpha
        1  b  2   beta

        ... or an array:
        >>> (s + Delta(q=np.linspace([1, 2], [10, 15], 3))).q
              0     1
        0   1.0   2.0
        1   5.5   8.5
        2  10.0  15.0

        ... or a dictionary:
        >>> (s + Delta(q={"a": [1,2,3], "b": [4,5,6]})).q
           a  b
        0  1  4
        1  2  5
        2  3  6

        ... or a list:
        >>> (s + Delta(q=[11, 12, 13])).q
            0
        0  11
        1  12
        2  13

        ... but not, for instance, a string:
        >>> (s + Delta(q="not compatible with pd.DataFrame")).q
        Traceback (most recent call last):
        ...
        ValueError: DataFrame constructor not properly called!

        Without a converter:
        >>> @dataclass(frozen=True)
        ... class CoerceStateDataFrameNoConverter(State):
        ...    r: pd.DataFrame = field(default_factory=pd.DataFrame, metadata={"delta": "replace"})

        ... there is no coercion – the object is passed unchanged
        >>> t = CoerceStateDataFrameNoConverter()
        >>> (t + Delta(r=np.linspace([1, 2], [10, 15], 3))).r
        array([[ 1. ,  2. ],
               [ 5.5,  8.5],
               [10. , 15. ]])


        A converter can cast from a DataFrame to a np.ndarray (with a single datatype),
        for instance:
        >>> @dataclass(frozen=True)
        ... class CoerceStateArray(State):
        ...    r: Optional[np.ndarray] = field(default=None,
        ...                            metadata={"delta": "replace",
        ...                                      "converter": np.asarray})

        Here we pass a dataframe, but expect a numpy array:
        >>> (CoerceStateArray() + Delta(r=pd.DataFrame([("a",1), ("b",2)], columns=list("xy")))).r
        array([['a', 1],
               ['b', 2]], dtype=object)

        We can define aliases which can transform between different potential field
        names.

    """

    def __add__(self, other: Union[Delta, Mapping]):
        updates = dict()
        other_fields_unused = list(other.keys())
        for self_field in fields(self):
            other_value, key = _get_value(self_field, other)
            if other_value is None:
                continue
            other_fields_unused.remove(key)

            self_field_key = self_field.name
            self_value = getattr(self, self_field_key)
            delta_behavior = self_field.metadata["delta"]

            if (constructor := self_field.metadata.get("converter", None)) is not None:
                coerced_other_value = constructor(other_value)
            else:
                coerced_other_value = other_value

            if delta_behavior == "extend":
                extended_value = _extend(self_value, coerced_other_value)
                updates[self_field_key] = extended_value
            elif delta_behavior == "append":
                appended_value = _append(self_value, coerced_other_value)
                updates[self_field_key] = appended_value
            elif delta_behavior == "replace":
                updates[self_field_key] = coerced_other_value
            else:
                raise NotImplementedError(
                    "delta_behaviour=`%s` not implemented" % delta_behavior
                )

        if len(other_fields_unused) > 0:
            warnings.warn(
                "These fields: %s could not be used to update %s, "
                "which has these fields & aliases: %s"
                % (
                    other_fields_unused,
                    type(self).__name__,
                    _get_field_names_and_aliases(self),
                ),
            )

        new = replace(self, **updates)
        return new

    def update(self, **kwargs):
        """
        Return a new version of the State with values updated.

        This is identical to adding a `Delta`.

        If you need to replace values, ignoring the State value aggregation rules,
        use `dataclasses.replace` instead.
        """
        return self + Delta(**kwargs)

    def copy(self):
        """
        Return a deepcopy of the State
        Examples:
            >>> @dataclass(frozen=True)
            ... class DfState(State):
            ...    q: pd.DataFrame = field(default_factory=pd.DataFrame,
            ...                            metadata={"delta": "replace",
            ...                                      "converter": pd.DataFrame})
            >>> data = pd.DataFrame({'x': [1, 2, 3]})
            >>> s_1 = DfState(q=data)
            >>> s_replace = replace(s_1)
            >>> s_copy = s_1.copy()

            The build in replace method doesn't create a deepcopy:
            >>> s_1.q is s_replace.q
            True
            >>> s_1.q['y'] = [1,2,3]
            >>> s_replace.q
               x  y
            0  1  1
            1  2  2
            2  3  3

            But this copy method does:
            >>> s_1.q is s_copy.q
            False
            >>> s_copy.q
               x
            0  1
            1  2
            2  3


        """
        # Create a dictionary to hold the field copies
        field_copies = {}

        # Iterate over all fields of the class
        for _field in fields(self):
            value = getattr(self, _field.name)
            # Use deepcopy to ensure that mutable fields are also copied
            field_copies[_field.name] = copy.deepcopy(value)

        # Use replace with **field_copies to create a new instance of the same class
        return replace(self, **field_copies)


def _get_value(f, other: Union[Delta, Mapping]):
    """
    Given a `State`'s `dataclasses.field` f, get a value from `other` and report its name.

    Returns: a tuple (the value, the key associated with that value)

    Examples:
        >>> from dataclasses import field, dataclass, fields
        >>> @dataclass
        ... class Example:
        ...     a: int = field()  # base case
        ...     b: List[int] = field(metadata={"aliases": {"ba": lambda b: [b]}})  # Single alias
        ...     c: List[int] = field(metadata={"aliases": {
        ...                                         "ca": lambda x: x,   # pass the value unchanged
        ...                                         "cb": lambda x: [x]  # wrap the value in a list
        ...        }})  # Multiple alias

        For a field with no aliases, we retrieve values with the base name:
        >>> f_a = fields(Example)[0]
        >>> _get_value(f_a, Delta(a=1))
        (1, 'a')

        ... and only the base name:
        >>> print(_get_value(f_a, Delta(b=2)))  # no match for b
        (None, None)

        Any other names are unimportant:
        >>> _get_value(f_a, Delta(b=2, a=1))
        (1, 'a')

        For fields with an alias, we retrieve values with the base name:
        >>> f_b = fields(Example)[1]
        >>> _get_value(f_b, Delta(b=[2]))
        ([2], 'b')

        ... or for the alias name, transformed by the alias lambda function:
        >>> _get_value(f_b, Delta(ba=21))
        ([21], 'ba')

        We preferentially get the base name, and then any aliases:
        >>> _get_value(f_b, Delta(b=2, ba=21))
        (2, 'b')

        ... , regardless of their order in the `Delta` object:
        >>> _get_value(f_b, Delta(ba=21, b=2))
        (2, 'b')

        Other names are ignored:
        >>> _get_value(f_b, Delta(a=1))
        (None, None)

        and the order of other names is unimportant:
        >>> _get_value(f_b, Delta(a=1, b=2))
        (2, 'b')

        For fields with multiple aliases, we retrieve values with the base name:
        >>> f_c = fields(Example)[2]
        >>> _get_value(f_c, Delta(c=[3]))
        ([3], 'c')

        ... for any alias:
        >>> _get_value(f_c, Delta(ca=31))
        (31, 'ca')

        ... transformed by the alias lambda function :
        >>> _get_value(f_c, Delta(cb=32))
        ([32], 'cb')

        ... and ignoring any other names:
        >>> print(_get_value(f_c, Delta(a=1)))
        (None, None)

        ... preferentially in the order base name, 1st alias, 2nd alias, ... nth alias:
        >>> _get_value(f_c, Delta(c=3, ca=31, cb=32))
        (3, 'c')

        >>> _get_value(f_c, Delta(ca=31, cb=32))
        (31, 'ca')

        >>> _get_value(f_c, Delta(cb=32))
        ([32], 'cb')

        >>> print(_get_value(f_c, Delta()))
        (None, None)

        This works with dict objects:
        >>> _get_value(f_a, dict(a=13))
        (13, 'a')

        ... with multiple keys:
        >>> _get_value(f_b, dict(a=13, b=24, c=35))
        (24, 'b')

        ... and with aliases:
        >>> _get_value(f_b, dict(ba=222))
        ([222], 'ba')

        This works with UserDicts:
        >>> class MyDelta(UserDict):
        ...     pass

        >>> _get_value(f_a, MyDelta(a=14))
        (14, 'a')

        ... with multiple keys:
        >>> _get_value(f_b, MyDelta(a=1, b=4, c=9))
        (4, 'b')

        ... and with aliases:
        >>> _get_value(f_b, MyDelta(ba=234))
        ([234], 'ba')

    """

    key = f.name
    aliases = f.metadata.get("aliases", {})

    value, used_key = None, None

    if key in other.keys():
        value = other[key]
        used_key = key
    elif aliases:  # ... is not an empty dict
        for alias_key, wrapping_function in aliases.items():
            if alias_key in other:
                value = wrapping_function(other[alias_key])
                used_key = alias_key
                break  # we only evaluate the first match

    return value, used_key


def _get_field_names_and_properties(s: State):
    """
    Get a list of field names and their aliases from a State object

    Args:
        s: a State object

    Returns: a list of field names and their aliases on `s`

    Examples:
        >>> from dataclasses import field
        >>> @dataclass(frozen=True)
        ... class SomeState(State):
        ...    l: List = field(default_factory=list)
        ...    m: List = field(default_factory=list)
        ...    @property
        ...    def both(self):
        ...        return self.l + self.m
        >>> _get_field_names_and_properties(SomeState())
        ['both', 'l', 'm']
    """
    result = _get_field_names_and_aliases(s)
    property_names = [
        attr
        for attr in dir(s)
        if isinstance(getattr(type(s), attr, None), property)
        and attr not in dir(object)
        and attr not in result
    ]
    return property_names + result


def _get_field_names_and_aliases(s: State):
    """
    Get a list of field names and their aliases from a State object

    Args:
        s: a State object

    Returns: a list of field names and their aliases on `s`

    Examples:
        >>> from dataclasses import field
        >>> @dataclass(frozen=True)
        ... class SomeState(State):
        ...    l: List = field(default_factory=list)
        ...    m: List = field(default_factory=list)
        >>> _get_field_names_and_aliases(SomeState())
        ['l', 'm']

        >>> @dataclass(frozen=True)
        ... class SomeStateWithAliases(State):
        ...    l: List = field(default_factory=list, metadata={"aliases": {"l1": None, "l2": None}})
        ...    m: List = field(default_factory=list, metadata={"aliases": {"m1": None}})
        >>> _get_field_names_and_aliases(SomeStateWithAliases())
        ['l', 'l1', 'l2', 'm', 'm1']

    """
    result = []

    for f in fields(s):
        name = f.name
        result.append(name)

        aliases = f.metadata.get("aliases", {})
        result.extend(aliases)

    return result


class Delta(UserDict, Generic[S]):
    """
    Represents a delta where the base object determines the extension behavior.

    Examples:
        >>> from dataclasses import dataclass

        First we define the dataclass to act as the basis:
        >>> from typing import Optional, List
        >>> @dataclass(frozen=True)
        ... class ListState:
        ...     l: Optional[List] = None
        ...     m: Optional[List] = None
        ...
    """

    pass


Result = Delta
"""`Result` is an alias for `Delta`."""


@singledispatch
def _extend(a, b):
    """
    Function to extend supported datatypes.

    """
    raise NotImplementedError("`_extend` not implemented for %s, %s" % (a, b))


@_extend.register(type(None))
def _extend_none(_, b):
    """
    Implementation of `_extend` to support None-types.

    Examples:
        >>> _extend(None, [])
        []

        >>> _extend(None, [3])
        [3]
    """
    return b


@_extend.register(list)
def _extend_list(a, b):
    """
    Implementation of `_extend` to support Lists.

    Examples:
        >>> _extend([], [])
        []

        >>> _extend([1,2], [3])
        [1, 2, 3]
    """
    return a + b


@_extend.register(pd.DataFrame)
def _extend_pd_dataframe(a, b):
    """
    Implementation of `_extend` to support DataFrames.

    Examples:
        >>> _extend(pd.DataFrame({"a": []}), pd.DataFrame({"a": []}))
        Empty DataFrame
        Columns: [a]
        Index: []

        >>> _extend(pd.DataFrame({"a": [1,2,3]}), pd.DataFrame({"a": [4,5,6]}))
           a
        0  1
        1  2
        2  3
        3  4
        4  5
        5  6
    """
    return pd.concat((a, b), ignore_index=True)


@_extend.register(np.ndarray)
def _extend_np_ndarray(a, b):
    """
    Implementation of `_extend` to support Numpy ndarrays.

    Examples:
        >>> _extend(np.array([(1,2,3), (4,5,6)]), np.array([(7,8,9)]))
        array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
    """
    return np.row_stack([a, b])


@_extend.register(dict)
def _extend_dict(a, b):
    """
    Implementation of `_extend` to support Dictionaries.

    Examples:
        >>> _extend({"a": "cats"}, {"b": "dogs"})
        {'a': 'cats', 'b': 'dogs'}
    """
    return dict(a, **b)


@_extend.register(Delta)
def _extend_delta(a, b):
    """
    Implementation of `_extend` to support Delta

    Examples:
        >>> _extend(Delta(conditions=[1,2]), Delta(conditions=[3,4]))
        {'conditions': [1, 2, 3, 4]}
    """
    res = copy.deepcopy(a)
    for key in b:
        if key in a:
            res[key] = _extend(a[key], b[key])
        else:
            res[key] = b[key]
    return Delta(res)


def _append(a: List[T], b: T) -> List[T]:
    """
    Function to create a new list with an item appended to it.

    Examples:
        Given a starting list `a_`:
        >>> a_ = [1, 2, 3]

        ... we can append a value:
        >>> _append(a_, 4)
        [1, 2, 3, 4]

        `a_` is unchanged
        >>> a_ == [1, 2, 3]
        True

        Why not just use `list.append`? `list.append` mutates `a` in place, which we can't allow
        in the AER cycle – parts of the cycle rely on purely functional code which doesn't
        (accidentally or intentionally) manipulate existing data.
        >>> list.append(a_, 4)  # not what we want
        >>> a_
        [1, 2, 3, 4]
    """
    return a + [b]


def inputs_from_state(f, input_mapping: Dict = {}):
    """Decorator to make target `f` into a function on a `State` and `**kwargs`.

    This wrapper makes it easier to pass arguments to a function from a State.

    It was inspired by the pytest "fixtures" mechanism.

    Args:
        f: a function with arguments that could be fields on a `State`
            and that returns a `Delta`.
        input_mapping: a dict that maps the input arguments of the function to the state fields

    Returns: a version of `f` which takes and returns `State` objects.

    Examples:
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class U(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We indicate the inputs required by the parameter names.
        The output must be (compatible with) a `Delta` object.
        >>> @inputs_from_state
        ... def experimentalist(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return new_conditions

        >>> experimentalist(U(conditions=[1,2,3,4]))
        [11, 12, 13, 14]

        >>> experimentalist(U(conditions=[101,102,103,104]))
        [111, 112, 113, 114]

        If our function uses a different keyword argument than the state field, we can use
        the input mapping:
        >>> def experimentalist_(X):
        ...     new_conditions = [x + 10 for x in X]
        ...     return new_conditions
        >>> experimentalist_on_state = inputs_from_state(experimentalist_, {'X': 'conditions'})
        >>> experimentalist_on_state(U(conditions=[1,2,3,4]))
        [11, 12, 13, 14]

        Both also work with the `State` as UserDict. Here, we use the StandardState
        >>> experimentalist(StandardState(conditions=[1, 2, 3, 4]))
        [11, 12, 13, 14]

        >>> experimentalist_on_state(StandardState(conditions=[1, 2, 3, 4]))
        [11, 12, 13, 14]

        A dictionary can be returned and used:
        >>> @inputs_from_state
        ... def returns_a_dictionary(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return {"conditions": new_conditions}
        >>> returns_a_dictionary(U(conditions=[2]))
        {'conditions': [12]}

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        >>> @inputs_from_state
        ... def theorist(experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs):
        ...     ivs = [vi.name for vi in variables.independent_variables]
        ...     dvs = [vi.name for vi in variables.dependent_variables]
        ...     X, y = experiment_data[ivs], experiment_data[dvs]
        ...     model = LinearRegression(fit_intercept=True).set_params(**kwargs).fit(X, y)
        ...     return model

        >>> @dataclass(frozen=True)
        ... class V(State):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experiment_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> v = V(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experiment_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )
        >>> model = theorist(v)
        >>> model.coef_, model.intercept_
        (array([[1.]]), array([2.]))

        Arguments from the state can be overridden by passing them in as keyword arguments (kwargs):
        >>> theorist(v, experiment_data=pd.DataFrame({"x": [0,1,2,3], "y": [12,13,14,15]}))\\
        ...     .intercept_
        array([12.])

        ... and other arguments supported by the inner function can also be passed
        (if and only if the inner function allows for and handles `**kwargs` arguments alongside
        the values from the state).
        >>> theorist(v, fit_intercept=False).intercept_
        0.0

        Any parameters not provided by the state must be provided by default values or by the
        caller. If the default is specified:
        >>> @inputs_from_state
        ... def experimentalist(conditions, offset=25):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return new_conditions

        ... then it need not be passed.
        >>> experimentalist(U(conditions=[1,2,3,4]))
        [26, 27, 28, 29]

        If a default isn't specified:
        >>> @inputs_from_state
        ... def experimentalist(conditions, offset):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return new_conditions

        ... then calling the experimentalist without it will throw an error:
        >>> experimentalist(U(conditions=[1,2,3,4]))
        Traceback (most recent call last):
        ...
        TypeError: experimentalist() missing 1 required positional argument: 'offset'

        ... which can be fixed by passing the argument as a keyword to the wrapped function.
        >>> experimentalist(U(conditions=[1,2,3,4]), offset=2)
        [3, 4, 5, 6]

        The same is true, if we don't provide a mapping for arguments:
        >>> def experimentalist_(X, offset):
        ...     new_conditions = [x + offset for x in X]
        ...     return new_conditions
        >>> experimentalist_on_state = inputs_from_state(experimentalist_, {'X': 'conditions'})
        >>> experimentalist_on_state(StandardState(conditions=[1,2,3,4]), offset=2)
        [3, 4, 5, 6]

        The state itself is passed through if the inner function requests the `state`:
        >>> @inputs_from_state
        ... def function_which_needs_whole_state(state, conditions):
        ...     print("Doing something on: ", state)
        ...     new_conditions = [c + 2 for c in conditions]
        ...     return new_conditions
        >>> function_which_needs_whole_state(U(conditions=[1,2,3,4]))
        Doing something on:  U(conditions=[1, 2, 3, 4])
        [3, 4, 5, 6]


        If the state has a @property alias, this is respected:
        >>> @inputs_from_state
        ... def function_which_needs_alias(alias):
        ...     print("Doing something with: ", alias)
        ...     new_conditions = [x + 2 for x in alias]
        ...     return new_conditions

        >>> @dataclass(frozen=True)
        ... class V(State):
        ...     conditions: List = field(default_factory=list, metadata={"delta": "extend"})
        ...
        ...     @property
        ...     def alias(self):
        ...         return self.conditions

        >>> function_which_needs_alias(V(conditions=[1,2,3,4]))
        Doing something with:  [1, 2, 3, 4]
        [3, 4, 5, 6]


    """
    # Get the set of parameter names from function f's signature

    reversed_mapping = {v: k for k, v in input_mapping.items()}

    parameters_ = set(inspect.signature(f).parameters.keys())
    missing_func_params = set(input_mapping.keys()).difference(parameters_)
    if missing_func_params:
        raise ValueError(
            f"The following keys in input_state_mapping are not parameters of the function: "
            f"{missing_func_params}"
        )

    @wraps(f)
    def _f(state_: State, /, **kwargs) -> State:
        # Get the parameters needed which are available from the state_.
        # All others must be provided as kwargs or default values on f.
        assert is_dataclass(state_) or isinstance(state_, UserDict)
        if is_dataclass(state_):
            from_state = parameters_.intersection(
                _get_field_names_and_properties(state_)
            )
            arguments_from_state = {k: getattr(state_, k) for k in from_state}
            from_state_input_mapping = {
                reversed_mapping.get(field_name, field_name): getattr(
                    state_, field_name
                )
                for field_name in _get_field_names_and_properties(state_)
                if reversed_mapping.get(field_name, field_name) in parameters_
            }
            arguments_from_state.update(from_state_input_mapping)
        elif isinstance(state_, UserDict):
            from_state = parameters_.intersection(set(state_.keys()))
            arguments_from_state = {k: state_[k] for k in from_state}
            from_state_input_mapping = {
                reversed_mapping.get(key, key): state_[key]
                for key in state_.keys()
                if reversed_mapping.get(key, key) in parameters_
            }
            arguments_from_state.update(from_state_input_mapping)
        if "state" in parameters_:
            arguments_from_state["state"] = state_
        arguments = dict(arguments_from_state, **kwargs)
        result = f(**arguments)
        return result

    return _f


def outputs_to_delta(*output: str):
    """
    Decorator factory to wrap outputs from a function as Deltas.

    Examples:
        >>> @outputs_to_delta("conditions")
        ... def add_five(x):
        ...     return [xi + 5 for xi in x]

        >>> add_five([1, 2, 3])
        {'conditions': [6, 7, 8]}

        >>> @outputs_to_delta("c")
        ... def add_six(conditions):
        ...     return [c + 5 for c in conditions]

        >>> add_six([1, 2, 3])
        {'c': [6, 7, 8]}

        >>> @outputs_to_delta("+1", "-1")
        ... def plus_minus_1(x):
        ...     a = [xi + 1 for xi in x]
        ...     b = [xi - 1 for xi in x]
        ...     return a, b

        >>> plus_minus_1([1, 2, 3])
        {'+1': [2, 3, 4], '-1': [0, 1, 2]}


        If the wrong number of values are specified for the return, then there might be errors.
        If multiple outputs are expected, but only a single output is returned, we get a warning:
        >>> @outputs_to_delta("1", "2")
        ... def returns_single_result_when_more_expected():
        ...     return "a"
        >>> returns_single_result_when_more_expected()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        AssertionError: function `<function returns_single_result_when_more_expected at 0x...>`
        has to return multiple values to match `('1', '2')`. Got `a` instead.

        If multiple outputs are expected, but the wrong number are returned, we get a warning:
        >>> @outputs_to_delta("1", "2", "3")
        ... def returns_wrong_number_of_results():
        ...     return "a", "b"
        >>> returns_wrong_number_of_results()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        AssertionError: function `<function returns_wrong_number_of_results at 0x...>`
        has to return exactly `3` values to match `('1', '2', '3')`. Got `('a', 'b')` instead.

        However, if a single output is expected, and multiple are returned, these are treated as
        a single object and no error occurs:
        >>> @outputs_to_delta("foo")
        ... def returns_a_tuple():
        ...     return "a", "b", "c"
        >>> returns_a_tuple()
        {'foo': ('a', 'b', 'c')}

        If we fail to specify output names, an error is returned immediately.
        >>> @outputs_to_delta()
        ... def decorator_missing_arguments():
        ...     return "a", "b", "c"
        Traceback (most recent call last):
        ...
        ValueError: `output` names must be specified.

    """

    def decorator(f):
        if len(output) == 0:
            raise ValueError("`output` names must be specified.")

        elif len(output) == 1:

            @wraps(f)
            def inner(*args, **kwargs):
                result = f(*args, **kwargs)
                delta = Delta(**{output[0]: result})
                return delta

        else:

            @wraps(f)
            def inner(*args, **kwargs):
                result = f(*args, **kwargs)
                assert isinstance(result, tuple), (
                    "function `%s` has to return multiple values "
                    "to match `%s`. Got `%s` instead." % (f, output, result)
                )
                assert len(output) == len(result), (
                    "function `%s` has to return "
                    "exactly `%s` values "
                    "to match `%s`. "
                    "Got `%s` instead."
                    "" % (f, len(output), output, result)
                )
                delta = Delta(**dict(zip(output, result)))
                return delta

        return inner

    return decorator


def delta_to_state(f):
    """Decorator to make `f` which takes a `State` and returns a `Delta` return an updated `State`.

    This wrapper handles adding a returned Delta to an input State object.

    Args:
        f: the function which returns a `Delta` object

    Returns: the function modified to return a State object

    Examples:
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class U(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We indicate the inputs required by the parameter names.
        The output must be (compatible with) a `Delta` object.
        >>> @delta_to_state
        ... @inputs_from_state
        ... def experimentalist(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return Delta(conditions=new_conditions)

        >>> experimentalist(U(conditions=[1,2,3,4]))
        U(conditions=[11, 12, 13, 14])

        >>> experimentalist(U(conditions=[101,102,103,104]))
        U(conditions=[111, 112, 113, 114])

        If the output of the function is not a `Delta` object (or something compatible with its
        interface), then an error is thrown.
        >>> @delta_to_state
        ... @inputs_from_state
        ... def returns_bare_conditions(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return new_conditions

        >>> returns_bare_conditions(U(conditions=[1])) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        AssertionError: Output of <function returns_bare_conditions at 0x...> must be a `Delta`,
        `UserDict`, or `dict`.

        A dictionary can be returned and used:
        >>> @delta_to_state
        ... @inputs_from_state
        ... def returns_a_dictionary(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return {"conditions": new_conditions}
        >>> returns_a_dictionary(U(conditions=[2]))
        U(conditions=[12])

        ... as can an object which subclasses UserDict (like `Delta`)
        >>> class MyDelta(UserDict):
        ...     pass
        >>> @delta_to_state
        ... @inputs_from_state
        ... def returns_a_userdict(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return MyDelta(conditions=new_conditions)
        >>> returns_a_userdict(U(conditions=[3]))
        U(conditions=[13])

        We recommend using the `Delta` object rather than a `UserDict` or `dict` as its
        functionality may be expanded in future.

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        >>> @delta_to_state
        ... @inputs_from_state
        ... def theorist(experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs):
        ...     ivs = [vi.name for vi in variables.independent_variables]
        ...     dvs = [vi.name for vi in variables.dependent_variables]
        ...     X, y = experiment_data[ivs], experiment_data[dvs]
        ...     new_model = LinearRegression(fit_intercept=True).set_params(**kwargs).fit(X, y)
        ...     return Delta(model=new_model)

        >>> @dataclass(frozen=True)
        ... class V(State):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experiment_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> v = V(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experiment_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )
        >>> v_prime = theorist(v)
        >>> v_prime.model.coef_, v_prime.model.intercept_
        (array([[1.]]), array([2.]))

        Arguments from the state can be overridden by passing them in as keyword arguments (kwargs):
        >>> theorist(v, experiment_data=pd.DataFrame({"x": [0,1,2,3], "y": [12,13,14,15]}))\\
        ...     .model.intercept_
        array([12.])

        ... and other arguments supported by the inner function can also be passed
        (if and only if the inner function allows for and handles `**kwargs` arguments alongside
        the values from the state).
        >>> theorist(v, fit_intercept=False).model.intercept_
        0.0

        Any parameters not provided by the state must be provided by default values or by the
        caller. If the default is specified:
        >>> @delta_to_state
        ... @inputs_from_state
        ... def experimentalist(conditions, offset=25):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then it need not be passed.
        >>> experimentalist(U(conditions=[1,2,3,4]))
        U(conditions=[26, 27, 28, 29])

        If a default isn't specified:
        >>> @delta_to_state
        ... @inputs_from_state
        ... def experimentalist(conditions, offset):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then calling the experimentalist without it will throw an error:
        >>> experimentalist(U(conditions=[1,2,3,4]))
        Traceback (most recent call last):
        ...
        TypeError: experimentalist() missing 1 required positional argument: 'offset'

        ... which can be fixed by passing the argument as a keyword to the wrapped function.
        >>> experimentalist(U(conditions=[1,2,3,4]), offset=2)
        U(conditions=[3, 4, 5, 6])

        The state itself is passed through if the inner function requests the `state`:
        >>> @delta_to_state
        ... @inputs_from_state
        ... def function_which_needs_whole_state(state, conditions):
        ...     print("Doing something on: ", state)
        ...     new_conditions = [c + 2 for c in conditions]
        ...     return Delta(conditions=new_conditions)
        >>> function_which_needs_whole_state(U(conditions=[1,2,3,4]))
        Doing something on:  U(conditions=[1, 2, 3, 4])
        U(conditions=[3, 4, 5, 6])

    """

    @wraps(f)
    def _f(state_: S, **kwargs) -> S:
        delta = f(state_, **kwargs)
        assert isinstance(delta, Mapping), (
            "Output of %s must be a `Delta`, `UserDict`, " "or `dict`." % f
        )
        new_state = state_ + delta
        return new_state

    return _f


def on_state(
    function: Optional[Callable] = None,
    input_mapping: Dict = {},
    output: Optional[Sequence[str]] = None,
):
    """Decorator (factory) to make target `function` into a function on a `State` and `**kwargs`.

    This combines the functionality of `outputs_to_delta` and `inputs_from_state`

    Args:
        function: the function to be wrapped
        output: list specifying State field names for the return values of `function`
        input_mapping: a dict that maps the keywords of the functions to the state fields

    Returns:

    Examples:
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class W(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We indicate the inputs required by the parameter names.
        >>> def add_ten(conditions):
        ...     return [c + 10 for c in conditions]
        >>> experimentalist = on_state(function=add_ten, output=["conditions"])

        >>> experimentalist(W(conditions=[1,2,3,4]))
        W(conditions=[11, 12, 13, 14])

        You can wrap functions which return a Delta object natively, by omitting the `output`
        argument:
        >>> @on_state()
        ... def add_five(conditions):
        ...     return Delta(conditions=[c + 5 for c in conditions])

        >>> add_five(W(conditions=[1, 2, 3, 4]))
        W(conditions=[6, 7, 8, 9])

        If you fail to declare outputs for a function which doesn't return a Delta:
        >>> @on_state()
        ... def missing_output_param(conditions):
        ...     return [c + 5 for c in conditions]

        ... an exception is raised:
        >>> missing_output_param(W(conditions=[1])) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        AssertionError: Output of <function missing_output_param at 0x...> must be a `Delta`,
        `UserDict`, or `dict`.

        You can use the @on_state(output=[...]) as a decorator:
        >>> @on_state(output=["conditions"])
        ... def add_six(conditions):
        ...     return [c + 6 for c in conditions]

        >>> add_six(W(conditions=[1, 2, 3, 4]))
        W(conditions=[7, 8, 9, 10])

        You can also declare an input-to-output mapping if the keyword arguments of the functions
        don't match the state fields:
        >>> @on_state(input_mapping={'X': 'conditions'}, output=["conditions"])
        ... def add_six(X):
        ...     return [x + 6 for x in X]

        >>> add_six(W(conditions=[1, 2, 3, 4]))
        W(conditions=[7, 8, 9, 10])

        This also works on the StandardState or other States that are defined as UserDicts:
        >>> add_six(StandardState(conditions=[1, 2, 3,4])).conditions
            0
        0   7
        1   8
        2   9
        3  10
    """

    def decorator(f):
        f_ = f
        if output is not None:
            f_ = outputs_to_delta(*output)(f_)
        f_ = inputs_from_state(f_, input_mapping)
        f_ = delta_to_state(f_)
        return f_

    if function is None:
        return decorator
    else:
        return decorator(function)


StateFunction = Callable[[State], State]


class StandardStateVariables(Enum):
    CONDITIONS = "conditions"
    EXPERIMENT_DATA = "experiment_data"
    MODELS = "models"
    VARIABLES = "variables"


@dataclass(frozen=True)
class StandardState(State):
    """
    Examples:
        The state can be initialized emtpy
        >>> from autora.variable import VariableCollection, Variable
        >>> s = StandardState()
        >>> s
        StandardState(variables=None, conditions=None, experiment_data=None, models=[])

        The `variables` can be updated using a `Delta`:
        >>> dv1 = Delta(variables=VariableCollection(independent_variables=[Variable("1")]))
        >>> s + dv1 # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StandardState(variables=VariableCollection(independent_variables=[Variable(name='1',...)

        ... and are replaced by each `Delta`:
        >>> dv2 = Delta(variables=VariableCollection(independent_variables=[Variable("2")]))
        >>> s + dv1 + dv2 # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StandardState(variables=VariableCollection(independent_variables=[Variable(name='2',...)

        The `conditions` can be updated using a `Delta`:
        >>> dc1 = Delta(conditions=pd.DataFrame({"x": [1, 2, 3]}))
        >>> (s + dc1).conditions
           x
        0  1
        1  2
        2  3

        ... and are replaced by each `Delta`:
        >>> dc2 = Delta(conditions=pd.DataFrame({"x": [4, 5]}))
        >>> (s + dc1 + dc2).conditions
           x
        0  4
        1  5

        Datatypes other than `pd.DataFrame` will be coerced into a `DataFrame` if possible.
        >>> import numpy as np
        >>> dc3 = Delta(conditions=np.core.records.fromrecords([(8, "h"), (9, "i")], names="n,c"))
        >>> (s + dc3).conditions
           n  c
        0  8  h
        1  9  i

        If they are passed without column names, no column names are inferred.
        This is to ensure that accidental mislabeling of columns cannot occur.
        Column names should usually be provided.
        >>> dc4 = Delta(conditions=[(6,), (7,)])
        >>> (s + dc4).conditions
           0
        0  6
        1  7

        Datatypes which are incompatible with a pd.DataFrame will throw an error:
        >>> s + Delta(conditions="not compatible with pd.DataFrame") \
# doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...

        Experiment data can be updated using a Delta:
        >>> ded1 = Delta(experiment_data=pd.DataFrame({"x": [1,2,3], "y": ["a", "b", "c"]}))
        >>> (s + ded1).experiment_data
           x  y
        0  1  a
        1  2  b
        2  3  c

        ... and are extended with each Delta:
        >>> ded2 = Delta(experiment_data=pd.DataFrame({"x": [4, 5, 6], "y": ["d", "e", "f"]}))
        >>> (s + ded1 + ded2).experiment_data
           x  y
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        4  5  e
        5  6  f

        If they are passed without column names, no column names are inferred.
        This is to ensure that accidental mislabeling of columns cannot occur.
        >>> ded3 = Delta(experiment_data=pd.DataFrame([(7, "g"), (8, "h")]))
        >>> (s + ded3).experiment_data
           0  1
        0  7  g
        1  8  h

        If there are already data present, the column names must match.
        >>> (s + ded2 + ded3).experiment_data
             x    y    0    1
        0  4.0    d  NaN  NaN
        1  5.0    e  NaN  NaN
        2  6.0    f  NaN  NaN
        3  NaN  NaN  7.0    g
        4  NaN  NaN  8.0    h

        `experiment_data` other than `pd.DataFrame` will be coerced into a `DataFrame` if possible.
        >>> import numpy as np
        >>> ded4 = Delta(
        ...     experiment_data=np.core.records.fromrecords([(1, "a"), (2, "b")], names=["x", "y"]))
        >>> (s + ded4).experiment_data
           x  y
        0  1  a
        1  2  b

        `experiment_data` which are incompatible with a pd.DataFrame will throw an error:
        >>> s + Delta(experiment_data="not compatible with pd.DataFrame") \
# doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...

        `models` can be updated using a Delta:
        >>> from sklearn.dummy import DummyClassifier
        >>> dm1 = Delta(models=[DummyClassifier(constant=1)])
        >>> dm2 = Delta(models=[DummyClassifier(constant=2), DummyClassifier(constant=3)])
        >>> (s + dm1).models
        [DummyClassifier(constant=1)]

        >>> (s + dm1 + dm2).models
        [DummyClassifier(constant=1), DummyClassifier(constant=2), DummyClassifier(constant=3)]

        The last model is available under the `model` property:
        >>> (s + dm1 + dm2).model
        DummyClassifier(constant=3)

        If there is no model, `None` is returned:
        >>> print(s.model)
        None

        `models` can also be updated using a Delta with a single `model`:
        >>> dm3 = Delta(model=DummyClassifier(constant=4))
        >>> (s + dm1 + dm3).model
        DummyClassifier(constant=4)


        We can use properties X, y, iv_names and dv_names as 'getters' ...
        >>> x_v = Variable('x')
        >>> y_v = Variable('y')
        >>> variables = VariableCollection(independent_variables=[x_v], dependent_variables=[y_v])
        >>> e_data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
        >>> s = StandardState(variables=variables, experiment_data=e_data)
        >>> @inputs_from_state
        ... def show_X(X):
        ...     return X
        >>> show_X(s)
           x
        0  1
        1  2
        2  3

        ... but nothing happens if we use them as `setters`:
        >>> @on_state
        ... def add_to_X(X):
        ...     res = X.copy()
        ...     res['x'] += 1
        ...     return Delta(X=res)
        >>> s = add_to_X(s)
        >>> s.X
           x
        0  1
        1  2
        2  3

        However, if the property has a deticated setter, we can still use them as getter:
        >>> s.model is None
        True

        >>> from sklearn.linear_model import LinearRegression
        >>> @on_state
        ... def add_model(_model):
        ...     return Delta(model=_model)
        >>> s = add_model(s, _model=LinearRegression())
        >>> s.models
        [LinearRegression()]

        >>> s.model
        LinearRegression()






    """

    variables: Optional[VariableCollection] = field(
        default=None, metadata={"delta": "replace"}
    )
    conditions: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "replace", "converter": pd.DataFrame}
    )
    experiment_data: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "extend", "converter": pd.DataFrame}
    )
    models: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend", "aliases": {"model": lambda model: [model]}},
    )

    @property
    def iv_names(self) -> List[str]:
        """
        Returns the names of the independent variables
        Examples:
            >>> s_empty = StandardState()
            >>> s_empty.iv_names
            []
            >>> from autora.variable import VariableCollection, Variable
            >>> iv_1 = Variable('variable_1')
            >>> iv_2 = Variable('variable_2')
            >>> variables = VariableCollection(independent_variables=[iv_1, iv_2])
            >>> s_variables = StandardState(variables=variables)
            >>> s_variables.iv_names
            ['variable_1', 'variable_2']
        """
        if not self.variables or not self.variables.independent_variables:
            return []
        return [iv.name for iv in self.variables.independent_variables]

    @property
    def dv_names(self) -> List[str]:
        """
        Returns the names of the independent variables
        Examples:
            >>> s_empty = StandardState()
            >>> s_empty.dv_names
            []
            >>> from autora.variable import VariableCollection, Variable
            >>> x = Variable('x')
            >>> y = Variable('y')
            >>> variables = VariableCollection(independent_variables=[x], dependent_variables=[y])
            >>> s_variables = StandardState(variables=variables)
            >>> s_variables.dv_names
            ['y']
        """
        if not self.variables or not self.variables.dependent_variables:
            return []
        return [dv.name for dv in self.variables.dependent_variables]

    @property
    def X(self) -> pd.DataFrame:
        """
        Returns the already observed conditions as a pd.DataFrame
        Examples:
            >>> s_empty = StandardState()
            >>> s_empty.X
            Empty DataFrame
            Columns: []
            Index: []
            >>> from autora.variable import VariableCollection, Variable
            >>> x = Variable('x')
            >>> y = Variable('y')
            >>> variables = VariableCollection(independent_variables=[x], dependent_variables=[y])
            >>> experiment_data = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [0, 2, 4, 6]})
            >>> s = StandardState(variables=variables, experiment_data=experiment_data)
            >>> s.X
               x
            0  0
            1  1
            2  2
            3  3
        """
        if self.experiment_data is None:
            return pd.DataFrame()
        return self.experiment_data[self.iv_names]

    @property
    def y(self) -> pd.DataFrame:
        """
        Returns the observations as a pd.DataFrame
        Examples:
            >>> s_empty = StandardState()
            >>> s_empty.y
            Empty DataFrame
            Columns: []
            Index: []
            >>> from autora.variable import VariableCollection, Variable
            >>> x = Variable('x')
            >>> y = Variable('y')
            >>> variables = VariableCollection(independent_variables=[x], dependent_variables=[y])
            >>> experiment_data = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [0, 2, 4, 6]})
            >>> s = StandardState(variables=variables, experiment_data=experiment_data)
            >>> s.y
               y
            0  0
            1  2
            2  4
            3  6
        """
        if self.experiment_data is None:
            return pd.DataFrame()
        return self.experiment_data[self.dv_names]

    @property
    def model(self):
        if len(self.models) == 0:
            return None
        # The property to access the backing field
        return self.models[-1]

    @model.setter
    def model(self, value):
        # Control the setting behavior
        self.models.append(value)


X = TypeVar("X")
Y = TypeVar("Y")
XY = TypeVar("XY")


def estimator_on_state(estimator: BaseEstimator) -> StateFunction:
    """
    Convert a scikit-learn compatible estimator into a function on a `State` object.

    Supports passing additional `**kwargs` which are used to update the estimator's params
    before fitting.

    Examples:
        Initialize a function which operates on the state, `state_fn` and runs a LinearRegression.
        >>> from sklearn.linear_model import LinearRegression
        >>> state_fn = estimator_on_state(LinearRegression())

        Define the state on which to operate (here an instance of the `StandardState`):
        >>> from autora.state import StandardState
        >>> from autora.variable import Variable, VariableCollection
        >>> import pandas as pd
        >>> s = StandardState(
        ...     variables=VariableCollection(
        ...         independent_variables=[Variable("x")],
        ...         dependent_variables=[Variable("y")]),
        ...     experiment_data=pd.DataFrame({"x": [1,2,3], "y":[3,6,9]})
        ... )

        Run the function, which fits the model and adds the result to the `StandardState` as the
        last entry in the .models list.
        >>> state_fn(s).models[-1].coef_
        array([[3.]])

    """

    @on_state()
    def theorist(
        experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs
    ):
        ivs = [v.name for v in variables.independent_variables]
        dvs = [v.name for v in variables.dependent_variables]
        X, y = experiment_data[ivs], experiment_data[dvs]
        new_model = estimator.set_params(**kwargs).fit(X, y)
        return Delta(models=[new_model])

    return theorist


def experiment_runner_on_state(f: Callable[[X], XY]) -> StateFunction:
    """Wrapper for experiment_runner of the form $f(x) \rarrow (x,y)$, where `f`
    returns both $x$ and $y$ values in a complete dataframe.

    Examples:
        The conditions are some x-values in a StandardState object:
        >>> from autora.state import StandardState
        >>> s = StandardState(conditions=pd.DataFrame({"x": [1, 2, 3]}))

        The function can be defined on a DataFrame, allowing the explicit inclusion of
        metadata like column names.
        >>> def x_to_xy_fn(c: pd.DataFrame) -> pd.Series:
        ...     result = c.assign(y=lambda df: 2 * df.x + 1)
        ...     return result

        We apply the wrapped function to `s` and look at the returned experiment_data:
        >>> experiment_runner_on_state(x_to_xy_fn)(s).experiment_data
           x  y
        0  1  3
        1  2  5
        2  3  7

        We can also define functions of several variables:
        >>> def xs_to_xy_fn(c: pd.DataFrame) -> pd.Series:
        ...     result = c.assign(y=c.x0 + c.x1)
        ...     return result

        With the relevant variables as conditions:
        >>> t = StandardState(conditions=pd.DataFrame({"x0": [1, 2, 3], "x1": [10, 20, 30]}))
        >>> experiment_runner_on_state(xs_to_xy_fn)(t).experiment_data
           x0  x1   y
        0   1  10  11
        1   2  20  22
        2   3  30  33

    """

    @on_state()
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        experiment_data = f(x, **kwargs)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def combined_functions_on_state(
    functions: List[Tuple[str, Callable]], output: Optional[Sequence[str]] = None
):
    """
    Decorator (factory) to make target list of `functions` into a function on a `State`.
    The resulting function uses a state field as input and combines the outputs of the
    `functions`.

    Args:
        functions: the list of functions to be wrapped
        output: list specifying State field names for the return values of `function`

    Examples:
        >>> @dataclass(frozen=True)
        ... class U(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})
        >>> identity = lambda conditions : conditions
        >>> double_conditions = combined_functions_on_state(
        ...     [('id_1', identity), ('id_2', identity)], output=['conditions'])
        >>> s = U([1, 2])
        >>> s_double = double_conditions(s)
        >>> s
        U(conditions=[1, 2])
        >>> s_double
        U(conditions=[1, 2, 1, 2])

        We can also pass parameters to the functions:
        >>> def multiply(conditions, multiplier):
        ...     return [el * multiplier for el in conditions]
        >>> double_and_triple = combined_functions_on_state(
        ...     [('doubler', multiply), ('tripler', multiply)], output=['conditions']
        ... )
        >>> s = U([1, 2])
        >>> s_double_triple = double_and_triple(
        ...     s, params={'doubler': {'multiplier': 2}, 'tripler': {'multiplier': 3}}
        ... )
        >>> s_double_triple
        U(conditions=[2, 4, 3, 6])

        If the functions return a Delta object, we don't need to provide an output argument:
        >>> def decrement(conditions, dec):
        ...     return Delta(conditions=[el-dec for el in conditions])
        >>> def increment(conditions, inc):
        ...     return Delta(conditions=[el+inc for el in conditions])
        >>> dec_and_inc = combined_functions_on_state(
        ...     [('decrement', decrement), ('increment', increment)])
        >>> s_dec_and_inc = dec_and_inc(
        ...     s, params={'decrement': {'dec': 10}, 'increment': {'inc': 2}})
        >>> s_dec_and_inc
        U(conditions=[-9, -8, 3, 4])

    """

    def f_(_state: State, params: Optional[Dict] = None):
        result_delta = None
        for name, function in functions:
            _f_input_from_state = inputs_from_state(function)
            if params is None:
                _params = {}
            else:
                _params = params
            if name in _params.keys():
                _delta = _f_input_from_state(_state, **_params[name])
            else:
                _delta = _f_input_from_state(_state)
            result_delta = _extend(result_delta, _delta)
        return result_delta

    if output:
        f_ = outputs_to_delta(*output)(f_)
    f_ = delta_to_state(f_)
    return f_
