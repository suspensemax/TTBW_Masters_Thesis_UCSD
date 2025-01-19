# lsdo_project_template
A template repository for LSDOlab projects

This repository serves as a template for all LSDOlab projects with regard to documentation, testing and hosting of open-source code.

*README.md file contains high-level information about your package: it's purpose, high-level instructions for installation and usage.*

# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/lsdo_project_template.git
```
If you want users to install a specific branch, run
```sh
pip install git+https://github.com/LSDOlab/lsdo_project_template.git@branch
```

**Enabled by**: Copying the `setup.py` file, changing your repository name and version, 
and adding all your dependencies into the list `install_requires`.

## Installation instructions for developers
To install `lsdo_project_template`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/lsdo_project_template.git
pip install -e ./lsdo_project_template
```
**Enabled by**: Copying the setup.py file, and changing your repository name and version.

# Documentation

## Writing

## Building

## Hosting

# Developer Guidelines

## Pull Requests
Before contributing to the `main` branch, make sure the following requirements are met:
- [ ] All existing tests have passed
- [ ] No local merge conflicts
- [ ] Code is commented
- [ ] Added tests for new features

Once you finish making changes on your branch, create a pull request.
To create the pull request, follow the steps:

1. Pull from `main` branch
```sh
git pull # Get most up-to-update version
```
2. Merge with main and check for conflicts
```sh
git merge main # merge with main locally on your branch to check for conflicts
```
3. Run tests
```sh
pytest # standard testing
pytest --disable-warnings # tests without displaying warnings
pytest -rP # tests while displaying print statements
```

4. Push changes

```sh
git push
```

5. Create pull request on GitHub with descriptions for changes.
 - Fill out pull request template
 - Assign reviewers (repo admin(s)?)

## Pull Request Review
When a pull request is submitted, reviews should be done in a timely manner (it will be time-consuming to remember what the code does if the code takes a week to be reviewed)

When reviewing a pull request, approve when:
 - Requirements above are met
   - Make sure code is commented cleanly 
 - GitHub actions tests pass
 - The code runs locally on your computer (run cases to try and break their code)

When rejecting a pull request, write a precise description of what needs to improved/fixed in the comment section.

## Tests
For test driven development, create tests before code implementation.
Templates from `tests/template.py` can be used. Create tests in `tests` directory and write test functions with `test_` prefix in test files.
