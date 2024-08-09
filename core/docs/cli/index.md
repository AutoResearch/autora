# Command Line Interface

Different parts of AutoRA experiments can require very different computational resources. For instance:

- The theorist and experimentalist might require training or use of neural networks, and benefit from high 
  performance computing (HPC) resources for short bursts – minutes or hours.
- The experiment runner might post an experiment using a service like "Prolific" and poll every few minutes for 
  hours, days or week until the experimental data are gathered.

Running the experiment runner with the same resources as the theorist and experimentalist in this case would be 
wasteful, and may be prohibitively expensive.

To solve this problem, AutoRA comes with a command line interface (CLI). This can be used with HPC schedulers like 
[SLURM](https://slurm.schedmd.com/) to run different steps in the cycle with different resources.

You can use the CLI if the following conditions are true:

1. Every part of `s` can be successfully [pickled](https://docs.python.org/3/library/pickle.html).
2. You can write each step of your experiment as a single importable function which operates on a state and returns 
   a state:
    ```python
    from example.lib import initial_state, experimentalist, experiment_runner, theorist
    s = initial_state()
    for i in range(3):
        s = experimentalist(s)
        s = experiment_runner(s)
        s = theorist(s)
    ```

Contents of this section:

- ["Basic Usage"](./basic-usage): Basic usage of the CLI
- ["Usage with Cylc workflow manager"](./cylc-pip): Example using the Cylc workflow manager (which 
  handles cyclical processes)
- ["Usage with Cylc workflow manager and Slurm"](./cylc-slurm-pip): Example using the Cylc 
  workflow manager and the SLURM scheduler with different resources for each step in the cycle.    
