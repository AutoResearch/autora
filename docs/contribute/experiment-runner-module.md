# Contribute an experimentalist

AutoRA experiment runners are designed to generate observations. They encompass a range of tools for conducting both synthetic and real experiments. By combining observations with corresponding experimental conditions, they provide inputs necessary for an AutoRA theorist to perform model discovery.
![Experimentalist Runner Module](../img/experiment_runner.png)

Experiment runners can be implemented as *synthetic runners*:
To contribute a *synthetic experiment runner* follow the [core](core.md) contribution guide.

Contributions may be complete experiment runners, which are functions that return observations, or tools that help automate experiments. Examples of such tools that are already implemented include a [recruitment manager](https://autoresearch.github.io/autora/user-guide/experiment-runners/recruitment-managers/prolific/) for recruiting participants on [Prolific](prolific.co) and an [experimentation manager](https://autoresearch.github.io/autora/user-guide/experiment-runners/experimentation-managers/firebase/) for executing online experiments with [Firebase](https://firebase.google.com/).

## Repository setup

For non-synthetic experiment runners, we recommend using the [cookiecutter template](https://github.com/AutoResearch/autora-template-cookiecutter) to set up
a repository for your experiment runner. Alternatively, you can use the 
[unguided template](https://github.com/AutoResearch/autora-template). If you choose the cookiecutter template, you can set up your repository using

```shell
cookiecutter https://github.com/AutoResearch/autora-template-cookiecutter
```

Make sure to select the `experiment runner` option when prompted. If you want to design a tool to recruit participants, choose the `recruitment manager` option. If you want to design a tool to conduct experiments, choose the `experimentation manager` option. You can also design a custom tool and name it yourself. You can skip all prompts pertaining to other modules 
(e.g., experimentalists) by pressing enter.

## Implementation

If you want to implement a complete experiment runner this should be a function that returns observations. But the experiment runner category also encompasses tools that help with automation of conducting experiments. For a more detailed overview, you can look at the [list of tools](https://autoresearch.github.io/autora/experiment-runner/) that are already implemented.


## Next steps: testing, documentation, publishing

For more information on how to test, document, and publish your experimentalist, please refer to the 
[general guideline for module contributions](module.md) . 
