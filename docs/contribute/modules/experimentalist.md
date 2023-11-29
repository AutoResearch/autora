# Contribute An Experimentalist

AutoRA experimentalists are meant to return novel experimental conditions based on prior experimental conditions, prior
observations, and/or prior models. Such conditions may serve as a basis for new, informative experiments conducted 
by an experiment runner. Experimentalists are generally implemented as functions.

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

Once you've created your repository, you can implement your experimentalist by editing the `init.py` file in 
``src/autora/experimentalist/name_of_your_experimentalist/``. You may also add additional files to this directory if needed. 
In the `init.py` file, implement your functions. Typicall names for these functions are `pool` (if you want a function that creates a pool of coinditions) or `sample` (if you want to sample from an existing conditions). Irrespective of the name the function should return a pandas Dataframe with the independent variables as columns.

Most experimentalists will use a subset of attributes from the [standard state](...) as input argumemts. These include the keyword arguments conditions, experiment_data, models, and variables. Additionally, many experimentalists will utilize an argument named num_samples, which determines the number of returned samples. The structure of a typical function body incorporating these elements is as follows:

```python
def sample(
        conditions: pd.DataFrame,
        experiment_data: pd.DataFrame,
        models: List,
        variables: VariableCollection,
        num_samples: n
        ) -> pd.DataFrame:

        return pd.DataFrame()
```

## Next Steps: Testing, Documentation, Publishing

For more information on how to test, document, and publish your experimentalist, please refer to the 
[general guideline for module contributions](index.md) . 
