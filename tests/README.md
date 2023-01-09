# Tests

## Seeding

Some testcases involve random numbers, including:
- All neural network models including DARTS, Bayesian DARTS
- All Tree models including BMS.

In order to avoid the testcases from running correctly sometimes, and incorrectly other times, we seed all the relevant random number generators for those testcases. To accomplish this, add the following pytest fixture to the test file and include it as required in the test functions:

```python
import random
import pytest
import torch

@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    random.seed(180)  # required for models which use the python `random` module, e.g. BMS
    torch.manual_seed(180)  # required for PyTorch models, e.g. DARTS
    return


def test_foo(seed):
    """ Test something. """
    
    # No need to use `seed` in the function body – adding it as an argument is sufficient
    
    ... # Run tests
```

The seed value should be consistent but not tuned to produce correct results. The integer `180` is used in many tests, inspired "180 George St., Providence, RI, USA", the office address for the Center for Computation and Visualization at Brown University, whose staff supported the development of the AutoRA package. Sensible alternatives are `42`, `31415926` and `2654435769`. See [https://en.wikipedia.org/wiki/Nothing-up-my-sleeve_number](https://en.wikipedia.org/wiki/Nothing-up-my-sleeve_number) for more inspiration.
