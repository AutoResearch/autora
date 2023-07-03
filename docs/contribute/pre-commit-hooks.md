# Pre-Commit Hooks

We use [`pre-commit`](https://pre-commit.com) to manage pre-commit hooks. 

Pre-commit hooks are programs which run before each git commit, and can read and potentially modify the files which are to be committed. 

We use pre-commit hooks to:
- enforce coding guidelines, including the `python` style-guide [PEP8](https://peps.python.org/pep-0008/) (`black` and `flake8`),
- to check the order of `import` statements (`isort`),
- to check the types of `python` objects (`mypy`).

The hooks and their settings are specified in the `.pre-commit-config.yaml` in each repository.

## Handling Pre-Commit Hook Errors

If your `git commit` fails because of the pre-commit hook, then you should:

1. Run the pre-commit hooks on the files which you have staged, by running the following command in your terminal: 
    ```zsh
    $ pre-commit run
    ```

2. Inspect the output. It might look like this:
   ```
   $ pre-commit run
   black....................Passed
   isort....................Passed
   flake8...................Passed
   mypy.....................Failed
   - hook id: mypy
   - exit code: 1
   
   example.py:33: error: Need type annotation for "data" (hint: "data: Dict[<type>, <type>] = ...")
   Found 1 errors in 1 files (checked 10 source files)
   ```
3. Fix any errors which are reported.
   **Important: Once you've changed the code, re-stage the files it to Git. 
   This might mean un-staging changes and then adding them again.**
4. If you have trouble:
   - Do a web-search to see if someone else had a similar error in the past.
   - Check that the tests you've written work correctly.
   - Check that there aren't any other obvious errors with the code.
   - If you've done all of that, and you still can't fix the problem, get help from someone else on the team.
5. Repeat 1-4 until all hooks return "passed", e.g.
   ```
   $ pre-commit run
   black....................Passed
   isort....................Passed
   flake8...................Passed
   mypy.....................Passed
   ```

It's easiest to solve these kinds of problems if you make small commits, often.  
