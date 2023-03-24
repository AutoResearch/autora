import asyncio
import pathlib
import pickle
import sys
import tempfile
from typing import Callable, TypeVar

State = TypeVar("State")


async def dispatch_to_subprocess(curried_online_executor: Callable[[], State]) -> State:
    """

    Args:
        curried_online_executor:

    Returns:


    Examples:
        >>> def f():
        ...     return 1
        >>> dispatch_to_subprocess(f)
        1

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_filepath = pathlib.Path(tmpdir, "function.pickle")
        output_filepath = pathlib.Path(tmpdir, "out.pickle")
        with open(input_filepath, "wb") as input_file:
            pickle.dump(curried_online_executor, input_file)
        code = f"""
import pickle;

with open({input_filepath}, 'rb') as input_file:
    curried_online_executor = pickle.load(input_file)

result = curried_online_executor()

with open({output_filepath}, 'wb') as output_file:
    pickle.dump(result, output_file)

"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE
        )

        print(proc)

        with open(output_filepath, "rb") as output_file:
            result = pickle.load(output_file)

        return result
