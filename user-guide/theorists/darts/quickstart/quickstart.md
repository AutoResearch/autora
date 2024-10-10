# Quickstart Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- `graphviz` (optional, required for computation graph visualizations): 
  [https://graphviz.org/download/](https://graphviz.org/download/)

Install DARTS as part of the `autora` package:

```shell
pip install -U "autora[theorist-darts]"
```

!!! success
    It is recommended to use a `python` environment manager like `virtualenv`.

Check your installation by running:
```shell
python -c "from autora.theorist.darts import DARTSRegressor; DARTSRegressor()"
```