import asyncio
import pathlib
import pickle
import sys
import tempfile
import time
from typing import TypeVar

Controller_ = TypeVar("Controller_")


async def dispatch_to_subprocess(controller: Controller_) -> Controller_:
    """

    Args:
        curried_online_executor:

    Returns:


    Examples:
        >>> from autora.controller import Controller
        >>> from
        >>> c = Controller(metadata=None, theories=[""], experiment_runner=)
        >>> cn = asyncio.run(dispatch_to_subprocess(c))
        <Process ...>
        >>> cn
        <autora...Controller object at 0x...>

    """
    with tempfile.TemporaryDirectory() as tmpdir:

        # Testing code --------------
        import subprocess

        subprocess.call(["open", "-a", "Finder", tmpdir])
        # ---------------------------

        input_filepath = pathlib.Path(tmpdir, "function.pickle")
        output_filepath = pathlib.Path(tmpdir, "out.pickle")
        with open(input_filepath, "wb") as input_file:
            pickle.dump(controller, input_file)

        code = f"""
import pickle
import time

with open("{input_filepath}", 'rb') as input_file:
    controller = pickle.load(input_file)

updated_controller = next(controller)

with open("{output_filepath}", 'wb') as output_file:
    pickle.dump(updated_controller, output_file)

"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE
        )

        print(proc)

        time.sleep(5)

        with open(output_filepath, "rb") as output_file:
            result = pickle.load(output_file)

    return result
