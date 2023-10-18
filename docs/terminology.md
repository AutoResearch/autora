# AutoRA Terminology

The following table includes naming conventions used throughout AutoRA.

| Term               | Description                                                                                                                                 | Relevant Modules                                |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| State              | Base object that has all relevant fields and which uses the Delta mechanism to modify those fields.                                         | Core                                            |
| StandardState      | An optional default State that has the following fields: variables, conditions, experiment_data, models.                                    | Core                                            |
| Variables          | A State field that holds experimental variables, which are defined according to name, type, units, allowed values, and range.               | Experimentalists, Experiment Runners, Theorists |
| VariableCollection | Immutable metadata about dependent variables, independent variables, and covariates.                                                        | Experimentalists, Experiment Runners, Theorists |
| Conditions         | A State field that defines what observations should be collected according to a specific combination of values of the independent variables | Experimentalists, Experiment Runners, Theorists |
| Experiment Data    | A State field that holds observations that correspond to the specified conditions.                                                          | Experiment Runners, Theorists                   |
| Model              | A State field that holds the the collection of best fit equations produced by theorists.                                                    | Theorists, Experimentalists                     |
| Components         | These are the distinct yet flexible capabilities of the AutoRA framework.                                                                   | Experimentalists, Experiment Runners, Theorists |
| Experimentalist    | A module that takes in models and outputs new conditions, which are intended to yield novel observations.                                   | Experimentalists                                |
| Theorist           | A module that takes in the full collection of conditions and observations and outputs equations that link the two (i.e., models)            | Theorists                                       |
| Experiment Runner  | A module that takes in conditions and collects corresponding observations.                                                                  | Experiment Runners                              |
| Wrapper            | Special functions that make the components of AutoRA able to operate on State objects.                                                      | Experimentalists, Experiment Runners, Theorists |
| Workflow           | A collection of tools that enable closed-loop empirical research with the AutoRA framework.                                                 | Experimentalists, Experiment Runners, Theorists |
| Cycle              | A workflow tool that allows AutoRA components to be chained together in serial loops.                                                       | Experimentalists, Experiment Runners, Theorists |
| Cylc               | A workflow engine for cycling systems that orchestrates distributed workflows of interdependent cycling tasks.                              | Experimentalists, Experiment Runners, Theorists |
