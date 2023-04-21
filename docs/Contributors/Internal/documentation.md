# Documentation

## Commenting code

To help users understand code better, and to make the documentation generation automatic, we have some standards for documenting code. The comments, docstrings, and the structure of the code itself are meant to make life easier for the reader. 
- If something important isn't _obvious_ from the code, then it should be _made_ obvious with a comment. 
- Conversely, if something _is_ obvious, then it doesn't need a comment.

These standards are inspired by John Ousterhout. *A Philosophy of Software Design.* Yaknyam Press, 2021. Chapter 12 – 14.

### Every public function, class and method has documentation

We include docstrings for all public functions, classes, and methods. These docstrings are meant to give a concise, high-level overview of **why** the function exists, **what** it is trying to do, and what is **important** about the code. (Details about **how** the code works are often better placed in detailed comments within the code.)

Every function, class or method has a one-line **high-level description** which clarifies its intent.   

The **meaning** and **type** of all the input and output parameters should be described.

There should be **examples** of how to use the function, class or method, with expected outputs, formatted as ["doctests"](https://docs.python.org/3/library/doctest.html). These should include normal cases for the function, but also include cases where it behaves unexpectedly or fails. 

We follow the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), as these are supported by the online documentation tool we use (see [Online Documentation](#online-documentation)).

A well documented function looks something like this:
```python
def first_order_linear(
    x: Union[float, np.ndarray], c: float, m: float
) -> Union[float, np.ndarray]:
    """
    Evaluate a first order linear model of the form y = m x + c.

    Arguments:
        x: input location(s) on the x-axis
        c: y-intercept of the linear model
        m: gradient of the linear model

    Returns:
        y: result y = m x + c, the same shape and type as x

    Examples:
        >>> first_order_linear(0. , 1. , 0. )
        1.0
        >>> first_order_linear(np.array([-1. , 0. , 1. ]), c=1.0, m=2.0)
        array([-1.,  1.,  3.])
    """
    y = m * x + c
    return y
```

*Pro-Tip: Write the docstring for your new high-level object before starting on the code. In particular, writing examples of how you expect it should be used can help clarify the right level of abstraction.*

## Online Documentation

Online Documentation is automatically generated using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) based on docstrings in files in the `autora/` directory. 

### Commands

Build and serve the documentation using the following commands:

* `poetry run mkdocs serve` - Start the live-reloading docs server.
* `poetry run mkdocs build` - Build the documentation site.
* `poetry run mkdocs gh-deploy` - Build the documentation and serve at https://AutoResearch.github.io/AutoRA/
* `poetry run mkdocs -h` - Print help message and exit.

### Documentation layout
```
mkdocs.yml    # The configuration file for the documentation.
docs/         # Directory for static pages to be included in the documentation.
    index.md  # The documentation homepage.
    ...       # Other markdown pages, images and other files.
autora/          # The directory containing the source code.
```
