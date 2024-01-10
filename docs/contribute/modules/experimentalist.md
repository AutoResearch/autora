# Contribute An Experimentalist

AutoRA experimentalists are meant to return novel experimental conditions based on prior experimental conditions, prior
observations, and/or prior models. Such conditions may serve as a basis for new, informative experiments conducted 
by an experiment runner. Experimentalists are generally implemented as functions that can be integrated into an 
[Experimentalist Pipeline](https://autoresearch.github.io/autora/core/docs/pipeline/Experimentalist%20Pipeline%20Examples/).

![Experimentalist Module](../../img/experimentalist.png)

## Repository Setup

We recommend using the [cookiecutter template](https://github.com/AutoResearch/autora-template-cookiecutter) to set up
a repository for your experimentalist. Alternatively, you can use the 
[unguided template](https://github.com/AutoResearch/autora-template). If you choose the cookiecutter template, you can set up your repository using

```shell
cookiecutter https://github.com/AutoResearch/autora-template-cookiecutter
```

Make sure to select the `experimentalist` option when prompted. You can skip all other prompts pertaining to other modules 
(e.g., experiment runners) by pressing enter.

## Implementation

For an experimentalist, you should implement a function that returns a set of experimental conditions. This set may be
a numpy array, iterator variable or other data format. 

!!! hint
    We generally **recommend using 2-dimensional numpy arrays as outputs** in which
    each row represents a set of experimental conditions. The columns of the array correspond to the independent variables.

Once you've created your repository, you can implement your experimentalist by editing the `init.py` file in 
``src/autora/experimentalist/name_of_your_experimentalist/``. 
You may also add additional files to this directory if needed. 
It is important that the `init.py` file contains a function called `name_of_your_experimentalist` 
which returns a set of experimental conditions (e.g., as a numpy array).

The following example ``init.py`` illustrates the implementation of a simple experimentalist
that uniformly samples without replacement from a pool of candidate conditions.

```python 
"""
Example Experimentalist
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
