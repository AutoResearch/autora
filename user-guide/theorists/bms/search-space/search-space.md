# Search Space

BMS searches the space of operations according to certain parameters to find the best model. As such, the search space is defined by the set of operations that can be applied in each computation step of the model. These operations are also referred to as *primitives*. We can select from the following set of primitives:

- **$\textit{constant}$**: The output of the computation $x_j$ is a constant parameter value $a$ where $a$ is a fitted float value.
- **\+**: The output of the computation $x_j$ is the sum over its two inputs $x_i, x_{ii}$: $x_j = x_i + x_{ii}$.
- **\-**: The output of the computation $x_j$ is the respective difference between its inputs $x_i, x_{ii}$: $x_j = x_i - x_{ii}$.
- **\***: The output of the computation $x_j$ is the product over its two inputs $x_i, x_{ii}$: $x_j = x_i * x_{ii}$.
- **/**: The output of the computation $x_j$ is the respective quotient between its inputs $x_i, x_{ii}$: $x_j = x_i / x_{ii}$.
- **abs**: The output of the computation $x_j$ is the absolute value of its input $x_i$: $x_j = |(x_i)|$.
- **relu**: The output of the computation $x_j$ is a rectified linear function applied to its input $x_i$: $x_j = \max(0, x_i)$.
- **exp**: The output of the computation $x_j$ is the natural exponential function applied to its input $x_i$: $x_j = \exp(x_i)$.
- **log**: The output of the computation $x_j$ is the natural logarithm function applied to its input $x_i$: $x_j = \log(x_i)$.
- **sig**: The output of the computation $x_j$ is a logistic function applied to its input $x_i$: $x_j = \frac{1}{1 + \exp(-b * x_i)}$.
- **fac**: The output of the computation $x_j$ is the generalized factorial function applied to its input $x_i$: $x_j = \Gamma(1 + x_i)$.
- **sqrt**: The output of the computation $x_j$ is the square root function applied to its input $x_i$: $x_j = \sqrt(x_i)$.
- **pow2**: The output of the computation $x_j$ is the square function applied to its input $x_i$: $x_j$ = $x_i^2$.
- **pow3**: The output of the computation $x_j$ is the cube function applied to its input $x_i$: $x_j$ = $x_i^3$.
- **sin**: The output of the computation $x_j$ is the sine function applied to its input $x_i$: $x_j = \sin(x_i)$.
- **sinh**: The output of the computation $x_j$ is the hyperbolic sine function applied to its input $x_i$: $x_j = \sinh(x_i)$.
- **cos**: The output of the computation $x_j$ is the cosine function applied to its input $x_i$: $x_j = \cos(x_i)$.
- **cosh**: The output of the computation $x_j$ is the hyperbolic cosine function applied to its input $x_i$: $x_j = \cosh(x_i)$.
- **tan**: The output of the computation $x_j$ is the tangent function applied to its input $x_i$: $x_j = \tan(x_i)$.
- **tanh**: The output of the computation $x_j$ is the hyperbolic tangent function applied to its input $x_i$: $x_j = \tanh(x_i)$.
- **\*\***: The output of the computation $x_j$ is the first input raised to the power of the second input $x_i,x_{ii}$: $x_j$ = $x_i^{x_{ii}}$.