import asyncio
import pathlib
import sys
import tempfile
from typing import TypeVar

import dill

Controller_ = TypeVar("Controller_")


async def dispatch_to_subprocess(controller: Controller_) -> Controller_:
    """

    Args:
        curried_online_executor:

    Returns:


    Examples:
        >>> from autora.controller import Controller
        >>> import numpy as np
        >>> def plus_1(x):
        ...     return x + 1
        >>> c = Controller(metadata=None, experiment_runner=plus_1)
        >>> c.state = c.state.update(conditions=[np.array([1,2,3])])
        >>> cn = asyncio.run(dispatch_to_subprocess(c))  # doctest: +ELLIPSIS
        >>> cn  # doctest: +ELLIPSIS
        <autora...Controller object at 0x...>

        >>> cn.state  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data={}, kind=ResultKind.PARAMS),
        Result(data=array([1, 2, 3]), kind=ResultKind.CONDITION),
        Result(data=array([[1, 2],
                           [2, 3],
                           [3, 4]]), kind=ResultKind.OBSERVATION)])

    """
    with tempfile.TemporaryDirectory() as tmpdir:

        input_filepath = pathlib.Path(tmpdir, "function.dill")
        output_filepath = pathlib.Path(tmpdir, "out.dill")
        with open(input_filepath, "wb") as input_file:
            dill.dump(controller, input_file)

        code = f"""
import dill

with open("{input_filepath}", 'rb') as input_file:
    controller = dill.load(input_file)

updated_controller = next(controller)

with open("{output_filepath}", 'wb') as output_file:
    dill.dump(updated_controller, output_file)

"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE
        )

        await proc.wait()

        with open(output_filepath, "rb") as output_file:
            result = dill.load(output_file)

    return result
