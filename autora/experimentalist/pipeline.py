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
        self.results: List[ExperimentalSequence] = []

    def __call__(self) -> ExperimentalSequence:
        self.results = []
        # Create pool
        if callable(self.pool):
            self.results.append(self.pool())
        else:
            self.results.append(self.pool)

        # Run filters
        for pipe in self.pipes:
            self.results.append(pipe(self.results[-1]))

        return self.results[-1]

    # May be more intuitive to run the pipeline with a named method. Similar to skl.
    def run(self):
        self.results = []
        # Create pool
        if callable(self.pool):
            self.results.append(self.pool())
        else:
            self.results.append(self.pool)

        # Run filters
        for pipe in self.pipes:
            self.results.append(pipe(self.results[-1]))

        return self.results[-1]
