# Inequality Experimentalist

The inequality experimentalist is a method used to compare experimental conditions and select new conditions based on a pairwise distance metric. Here's how it works:

Given:
- Existing experimental conditions represented by $\vec{x}$ in the set $X$.
- Candidate experimental conditions represented by $\vec{x}'$ in the set $X'$.
- A pairwise distance metric $d(\vec{x}, \vec{x}')$ that calculates the distance between $\vec{x}$ and $\vec{x}'$.
- A threshold value (default = 0) that determines the maximum allowable distance for two conditions to be considered equal.
- A number $n$ of conditions to sample.

The inequality experimentalist operates as follows:

1. For each candidate condition $\vec{x}'$ in $X'$ calculate an $inequality$ $score$:
2. Calculate the distances $d(\vec{x}, \vec{x}')$ between $\vec{x}$ and $\vec{x}'$ using the pairwise distance metric for all $\vec{x}$ in $X$.
3. If $d(\vec{x}, \vec{x}')$ is greater than the threshold:
     - Consider $\vec{x}'$ as different from the existing condition $\vec{x}$.
     - add 1 to the $inequality$ $score$ for $\vec{x'}$
4.  If $d(\vec{x}, \vec{x}')$ is less than the threshold:
     - Consider $\vec{x}'$ as equal to the existing condition $\vec{x}$.
     - Do not add 1 to the score for $\vec{x'}$

The $n$ $\vec{x'}$ with the highest $inequality$ $scores$ are chosen as new conditions.
