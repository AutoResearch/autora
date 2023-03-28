# Description

Provide a high-level summary of the change and which issue (either on Github or Notion) it relates to. 
Include relevant motivation and context.

*If the issue is on Github, simply link to it using the '#' symbol; otherwise, provide a notion page link)*:
- resolves #(issue)

# Type of change

*Delete as appropriate:*
- **fix**: A bug fix
- **feat**: A new feature
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **docs**: Documentation only changes
- **test**: Adding missing tests or correcting existing tests
- **perf**: A code change that improves performance
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **ci**: Changes to our CI configuration files and scripts (i.e., those in the .github directory)
- **build**: Changes that affect the build system or external dependencies (i.e., poetry, conda)

If this is a breaking change (which might apply to any of the other kinds of change), note that below, otherwise delete 
this section:
- **BREAKING_CHANGE**: {description}

> Update the title of the PR to follow the convention:
> `{type}{! if BREAKING_CHANGE}: {short description}`, e.g. 
> - "fix: add missing foo in module bar" 
> - "feat: new theorist abc",
> - "feat!: replace module x with module y"
> - "docs: add notebooks on theorist z" 
> 
> ... then delete this block.

# Features (Optional)
- bulleted list of features, describing the code contributions in this pull request

# Questions (Optional)
- potential questions for the code reviewer (e.g., if you want feedback on a certain method or class; any feedback required from other code contributors with their mentioning)

# Remarks (Optional)
- any additional remarks for the code reviewer
- if your remarks or questions pertain to individual lines in the code, you may also include them as comments in the code review window

# Screenshots (Optional)
If your PR is related to UI change or anything visual, feel free to include screenshots to give the reviewer straightforward understandings on your changes.
