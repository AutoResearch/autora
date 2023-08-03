# Experimentalist Overview

The primary goal of an experimentalist is to identify experiments that yield 
scientific merit. AutoRA implements techniques for automating the identification 
of novel experiments.

An experiment consists of a series of **experimental conditions** $\vec{x} \in X$. 
The experimental variables manipulated in each experimental condition 
are defined as **factors**, and the values of each variable to be sampled 
in the experiment are defined as **levels** of the corresponding **factors**. 
As an example, consider a visual discrimination tasks in which participants are presented
with two lines of different lengths, and are asked to indicate which line is longer.
There are two factors in this experiment: the length of the first line and 
the length of the second line. Instances of the two line lengths 
(e.g., 2.0 cm for the first line and 2.1 cm for the second line) 
can be considered levels of the two factors, respectively. Thus, *an experimental condition is a vector of values that
corresponds to a specific combination of experiment levels $x_i$, 
each of which is an instance of an experiment factor.*

Experimentalists in AutoRA serve to identify novel 
experimental conditions $\vec{x} \in X$, where $x_i$ corresponds 
to the level of an experimental factor $i$.

![Overview](../img/experimentalist.png)

Experimentalists may use information about candidate models $M$ obtained from a theorist, 
experimental conditions that have already been probed $\vec{x}' \in X'$, or 
respective dependent measures $\vec{y}' \in Y'$. The following table includes the experimentalists currently implemented
 in AutoRA.

| Name               | Links                                                                                                                                                                                          | Function                                                                                                                      | Arguments   |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-------------|
| Random             | [Package](https://pypi.org/project/autora-core/), [Docs](https://autoresearch.github.io/autora/core/docs/experimentalists/sampler/random/)                                                     | $\vec{x_i} \sim U[a_i,b_i]$                                                                                                   |             |
| Novelty            | [Package](https://pypi.org/project/autora-experimentalist-sampler-novelty/), [Docs](https://autoresearch.github.io/autora/user-guide/experimentalists/samplers/novelty/)                       | $\underset{\vec{x}}{\arg\max}~\min(d(\vec{x}, \vec{x}'))$                                                                     | $X'$        |
| Uncertainty        | [Package](https://pypi.org/project/autora-experimentalist-sampler-uncertainty/), [Docs](https://autoresearch.github.io/autora/user-guide/experimentalists/samplers/uncertainty/)               | $\underset{\vec{x}}{\arg\max}~1 - P_M(\hat{y}^*, \vec{x})$, $\hat{y}^* = \underset{\hat{y}}{\arg\max}~P_M(\hat{y}_i \vec{x})$ | $M$         |
| Model Disagreement | [Package](https://pypi.org/project/autora-experimentalist-sampler-model-disagreement/), [Docs](https://autoresearch.github.io/autora/user-guide/experimentalists/samplers/model-disagreement/) | $\underset{\vec{x}}{\arg\max}~(P_{M_1}(\hat{y}, \vec{x}) - P_{M_2}(\hat{y} \vec{x}))^2$                                       | $M$         |
| Falsification      | [Package](https://pypi.org/project/autora-experimentalist-falsification/), [Docs](https://autoresearch.github.io/autora/falsification/docs/sampler/)                                           | $\underset{\vec{x}}{\arg\max}~\hat{\mathcal{L}}(M,X',Y',\vec{x})$                                                             | $M, X', Y'$ |
| Mixture            | [Package](https://pypi.org/project/mixture-experimentalist/), [Docs](https://autoresearch.github.io/autora/user-guide/experimentalists/samplers/mixture/)                                      | [ ]                                                                                                                           | [ ]         |
| Nearest Value      | [Package](https://pypi.org/project/autora-experimentalist-sampler-nearest-value/), [Docs](https://autoresearch.github.io/autora/user-guide/experimentalists/samplers/nearest-value/)           | [ ]                                                                                                                           | [ ]         |
| Leverage           |                                                                                                                                                                                                | [ ]                                                                                                                           | [ ]         |
| Inequality         | [Package](https://pypi.org/project/autora-experimentalist-sampler-inequality/), [Docs](https://autoresearch.github.io/autora/user-guide/experimentalists/samplers/inequality/)                 | [ ]                                                                                                                           | [ ]         |
| Assumption         |                                                                                                                                                                                                | [ ]                                                                                                                           | [ ]         |
