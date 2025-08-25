import logging
from functools import wraps
from typing import Callable

_logger = logging.getLogger(__name__)


def deprecate(
    f: Callable,
    message: str,
    callback: Callable = _logger.warning,
):
    """
    Wrapper to make function aliases which print a warning that a name is an alias.

    Args:
        f: the function to be aliased
        message: the message to be emitted when the deprecated code is used
        callback: a function to call to handle the warning message

    Examples:
        >>> def original():
        ...     return 1
        >>> deprecated = deprecate(original, "`original` is deprecated.")

        The original function is unaffected:
        >>> original()
        1

        The aliased function works the same way, but also emits a warning.
        >>> deprecated()  # doctest: +SKIP
        `original` is deprecated.
        1

        You can also set a custom callback instead of the default "warning":
        >>> a0 = deprecate(original, "`original` is deprecated.", callback=print)
        >>> a0()
        `original` is deprecated.
        1
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        callback(message)
        return f(*args, **kwds)

    return wrapper


def deprecated_alias(
    f: Callable, alias_name: str, callback: Callable = _logger.warning
):
    """
    Wrapper to make function aliases which print a warning that a name is an alias.

    Args:
        f: the function to be aliased
        alias_name: the name under which the function is aliased,
            like `foo = deprecated_alias(bar, "foo")
        callback: a function to call to handle the warning message

    Examples:
        >>> def original():
        ...     return 1
        >>> alias = deprecated_alias(original, "alias")

        The original function is unaffected:
        >>> original()
        1

        The aliased function works the same way, but also emits a warning.
        >>> alias()  # doctest: +SKIP
        Use `original` instead. `alias` is deprecated.
        1

        You can also set a custom callback instead of the default "warning":
        >>> a0 = deprecated_alias(original, "a0", callback=print)
        >>> a0()
        Use `original` instead. `a0` is deprecated.
        1

        The callback is given a single argument, the warning string.
        You can replace it if you like:
        >>> a0 = deprecated_alias(original, "a0", callback=lambda _: print("alternative message"))
        >>> a0()
        alternative message
        1
    """
    message = "Use `%s` instead. `%s` is deprecated." % (f.__name__, alias_name)
    wrapped = deprecate(f=f, message=message, callback=callback)
    return wrapped
