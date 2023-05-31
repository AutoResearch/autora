# Experimentalist Overview

The primary goal of an experimentalist is to identify experiments that yield 
scientific merit. AutoRA implements techniques for automating the identification 
of novel experiments.

An experiment consists of a series of **experiment conditions** $\vec{x} \in X$. 
The experimental variables manipulated in each experiment condition 
are defined as **factors**, and the values of each variable to be sampled 
in the experiment are defined as **levels** of the corresponding **factors**. 
As an example, consider a visual discrimination tasks in which participants are presented
with two lines of different lengths, and are asked to indicate which line is longer.
There are two factors in this experiment: the length of the first line and 
the length of the second line. Instances of the two line lengths 
(e.g., 2.0 cm for the first line and 2.1 cm for the sceond line) 
can be considered levels of the two factors, respectively. Thus, *an experiment condition is a vector of values that
corresponds to a specific combination of experiment levels $x_i$, 
each of which is an instance of an experiment factor.*

Experimentalists in AutoRA serve to identify novel 
experiment conditions $\vec{x} \in X$, where $x_i$ corresponds 
to the level of an experimental factor $i$.

![Overview](../img/experimentalist.png)

Experimentalists may use information about candidate models $M$ obtained from a theorist, 
experiment conditions that have already been probed $\vec{x}' \in X'$, or 
respective dependent measures $\vec{y}' \in Y'$. The following table includes the experimentalists currently implemented
 in AutoRA.

| Experimentalist  | Function                                                                                                                      | Arguments  |
|------------------|-------------------------------------------------------------------------------------------------------------------------------|------------|
| Random           | $\vec{x_i} \sim U[a_i,b_i]$                                                                                                   |            |
| Novelty          | $\underset{\vec{x}}{\arg\max}~\min(d(\vec{x}, \vec{x}'))$                                                                     | $X'$       |
| Least Confident  | $\underset{\vec{x}}{\arg\max}~1 - P_M(\hat{y}^*, \vec{x})$, $\hat{y}^* = \underset{\hat{y}}{\arg\max}~P_M(\hat{y}_i \vec{x})$ | $M$        |
| Model Comparison | $\underset{\vec{x}}{\argmax}~(P_{M_1}(\hat{y}, \vec{x}) - P_{M_2}(\hat{y} \vec{x}))^2$                                        | $M$        |
| Falsification    | $\underset{\vec{x}}{\argmax}~\hat{\mathcal{L}}(M,X',Y',\vec{x})$                                                              | $M, X', Y'$ |



