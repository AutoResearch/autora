# AutoRA Terminology

The following table includes naming conventions used throughout AutoRA.

## [ ]

| Term               | Description                                                                                                                                 | Relevant Modules                                |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| State              | Base object that has all relevant fields and which uses the Delta mechanism to modify those fields.                                         | Core                                            |
| StandardState      | An optional default State that has the following fields: variables, conditions, experiment_data, models.                                    | Core                                            |
| Variable           | Definition of an experimental variable, including its name, type, units, allowed values, and range.                                         | Experimentalists, Experiment Runners, Theorists |
| VariableCollection | Immutable metadata about dependent variables, independent variables, and covariates.                                                        | Experimentalists, Experiment Runners, Theorists |
| Conditions         | A State field that defines what observations should be collected according to a specific combination of values of the independent variables | Experimentalists, Experiment Runners, Theorists |
| Experiment Data    | abc xyz                                                                                                                                     | Experiment Runners, Theorists                   |
| Model              | abc xyz                                                                                                                                     | Theorists, Experimentalists                     |
| Components         | abc xyz                                                                                                                                     | Experimentalists, Experiment Runners, Theorists |
| Experimentalist    | abc xyz                                                                                                                                     | Experimentalists                                |
| Theorist           | abc xyz                                                                                                                                     | Theorists                                       |
| Experiment Runner  | abc xyz                                                                                                                                     | Experiment Runners                              |
| Wrapper            | abc xyz                                                                                                                                     | Experimentalists, Experiment Runners, Theorists |
| Workflow           | abc xyz                                                                                                                                     | Experimentalists, Experiment Runners, Theorists |
| Cycle              | abc xyz                                                                                                                                     | Experimentalists, Experiment Runners, Theorists |
| Cylc               | abc xyz                                                                                                                                     | Experimentalists, Experiment Runners, Theorists |
