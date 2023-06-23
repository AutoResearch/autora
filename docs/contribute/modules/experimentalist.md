# Contribute An Experimentalist

AutoRA experimentalists are meant to return novel experimental conditions based on prior experimental conditions, prior
observations, and/or prior models. Such conditions may serve as a basis for new, informative experiments conducted 
by an experiment runner. Experimentalists are generally implemented as functions that can be integrated into an 
[Experimentalist Pipeline](https://autoresearch.github.io/autora/core/docs/pipeline/Experimentalist%20Pipeline%20Examples/).

![Experimentalist Module](../../img/experimentalist.png)

Experimentalists can be implemented as *poolers* or as *samplers*.
- **Poolers** return a pool of candidate experimental conditions, which can be passed to a sampler that selects
  a subset of conditions from the pool to be used in the next experiment.
- **Samplers** directly return a subset of experimental conditions from a pool of candidate experimental conditions that already exist.

## Repository Setup

We recommend using the [cookiecutter template](https://github.com/AutoResearch/autora-template-cookiecutter) to set up
a repository for your experimentalist. Alternatively, you can use the 
[unguided template](https://github.com/AutoResearch/autora-template). If you choose the cookiecutter template, you can set up your repository using

```shell
cookiecutter https://github.com/AutoResearch/autora-template-cookiecutter
```

Make sure to select the `experimentalist` option when prompted. You may also select whether you want to implement an experimentalist as a sampler, pooler, or custom function. You can skip all other prompts pertaining to other modules 
(e.g., experiment runners) by pressing enter.

## Implementation

Irrespective of whether you are implementing a pooler or a sampler, 
you should implement a function that returns a set of experimental conditions. This set may be
a numpy array, iterator variable or other data format. 

!!! hint
    We generally **recommend using 2-dimensional numpy arrays as outputs** in which
    each row represents a set of experimental conditions. The columns of the array correspond to the independent variables.

### Implementing Poolers

Once you've created your repository, you can implement your experimentalist pooler by editing the `init.py` file in 
``src/autora/experimentalist/pooler/name_of_your_experimentalist/``. 
You may also add additional files to this directory if needed. 
It is important that the `init.py` file contains a function called `name_of_your_experimentalist` 
which returns a pool of experimental conditions (e.g., as an iterator object or numpy array).

The following example ``init.py`` illustrates the implementation of a simple experimentalist pooler
that generates a grid of samples within the specified bounds of each independent variable (IV):

```python 

"""
Example Experimentalist Pooler
"""

from itertools import product
from typing import List
from autora.variable import IV


def grid_pool(ivs: List[IV]):
    """
    Creates exhaustive pool from discrete values using a Cartesian product of sets

    Arguments:
        ivs {List[IV]}:  List of independent variables

    Returns:
        pool: An iterator over all possible combinations of IV values
    """

    l_iv_values = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"gridsearch_pool only supports independent variables with discrete allowed values, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)

    # Return Cartesian product of all IV values
    return product(*l_iv_values)


```

### Implementing Samplers

Once you've created your repository, you can implement your experimentalist sampler by editing the `init.py` file in 
``src/autora/experimentalist/sampler/name_of_your_experimentalist/``. 
You may also add additional files to this directory if needed. 
It is important that the `init.py` file contains a function called `name_of_your_experimentalist` 
which returns a set of experimental conditions (e.g., as a numpy array) given a pool of candidate experimental conditions.

The following example ``init.py`` illustrates the implementation of a simple experimentalist sampler
that uniformly samples without replacement from a pool of candidate conditions.

```python 
"""
Example Experimentalist Sampler
"""

import random
from typing import Iterable, Sequence, Union

random_sample(conditions: Union[Iterable, Sequence], n: int = 1):
    """
    Uniform random sampling without replacement from a pool of conditions.
    Args:
        conditions: Pool of conditions
        n: number of samples to collect

    Returns: Sampled pool

    """

    if isinstance(conditions, Iterable):
        conditions = list(conditions)
    random.shuffle(conditions)
    samples = conditions[0:n]

    return samples
```


## Next Steps: Testing, Documentation, Publishing

For more information on how to test, document, and publish your experimentalist, please refer to the 
[general guideline for module contributions](index.md) . 
