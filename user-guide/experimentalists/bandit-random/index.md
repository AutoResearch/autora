# bandit-random

This package provides functions to randomly sample a list of

- probability sequences
- reward sequences

## Probability sequence

A probability sequence is a sequence of vectors with elements in the range between 0 and 1:

Example for a probability function that can be used in a 3-arm bandit task:

```
[[0, 1., .3], [.6, .2, .8], ...]
```

## Reward sequence

A reward sequences uses the probabilities to generate a sequence with elements of either 0 or 1:

Example for a probability function that can be used in a 3-arm bandit task:

```
[[0, 1, 0], [1, 0, 1], ...]
```

The probability sequence can be created by specifying an initial probability for each element and a
drift:

For example:

```
initial_proba = [0, .5, 1.]
drift = [.1, 0., -.1]
...
sequence = [[0, .5, 1.], [.1, .5, .9], [.2, .5, .8], [.3, .5, .7]...]
```

Instead of fixed values for the initial probability and the drift, we can also use ranges. In that
case the values are randomly sampled from the range.

```
initial_proa = [[0, .3], [.4, .7], [.8, 1.]]
drift = [[0, .1], [.1, .2], [.2, .3]]
```




