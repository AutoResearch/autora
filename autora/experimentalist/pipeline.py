from typing import List, Protocol, Union


# TODO: Change this to be an np.ndarray
class ExperimentalSequence:
    pass


class Pool(Protocol):
    def __call__(self) -> ExperimentalSequence:
        ...


class Pipe(Protocol):
    def __call__(self, ex: ExperimentalSequence) -> ExperimentalSequence:
        ...


PipelineElement = Union[Pool, Pipe]


class Pipeline:
    def __init__(self, pool: Pool, *pipes: Pipe):
        self.pool = pool
        self.pipes = pipes

    def __call__(self) -> ExperimentalSequence:
        results: List[ExperimentalSequence] = []
        results.append(self.pool())
        for pipe in self.pipes:
            results.append(pipe(results[-1]))

        return results[-1]
