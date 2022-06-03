# Autonomous Empirical Research
Autonomous Empirical Research is an open source AI-based system for automating each aspect empirical research in the behavioral sciences, from the construction of a scientific hypothesis to conducting novel experiments.

# Contributors (Alphabetic Order)
Ben Andrew, Hannah Even, Ioana Marinescu, Sebastian Musslick, Sida Li

# Getting started (development)

## Running the tests
You should run the tests before you commit code to the repository, to ensure that you've not broken anything. 

The unit tests can be run as follows (starting in the root directory of the repository):

```zsh
$ python -m unittest
```

## Pre-Commit Hooks

We use [pre-commit](https://pre-commit.com) to manage pre-commit hooks. 
Pre-commit hooks are programs which run before each git commit and which check that the files to be committed: 
- are correctly formatted and 
- have no obvious coding errors. 
Pre-commit hooks are intended to enforce coding guidelines, including the Python style-guide [PEP8](https://peps.python.org/pep-0008/). 

The hooks and their settings are specified in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml).

### Handling Pre-Commit Hook Errors

If your `git commit` fails because of the pre-commit hook, then you should:

1. run the following command in your terminal: 
    ```zsh
    $ pre-commit run
    ```

2. Inspect the output. It might look like this:
   ```
   $ pre-commit run
   black....................................................................Passed
   isort....................................................................Passed
   flake8...................................................................Passed
   mypy.....................................................................Failed
   - hook id: mypy
   - exit code: 1
   
   example.py:33: error: Need type annotation for "data" (hint: "data: Dict[<type>, <type>] = ...")
   Found 1 errors in 1 files (checked 10 source files)
   ```
3. Fix any errors which are reported. If you have trouble:
   - Do a web-search to see if someone else had a similar error in the past.
   - Check that the tests you've written work correctly.
   - Check that there aren't any other obvious errors with the code.
   - If you've done all of that, and you still can't fix the problem, get help from someone else on the team.
4. Repeat 1-3 until all hooks return "passed", e.g.
   ```
   $ pre-commit run
   black....................................................................Passed
   isort....................................................................Passed
   flake8...................................................................Passed
   mypy.....................................................................Passed
   ```

It's easiest to solve these kinds of problems if you make small commits, often.  