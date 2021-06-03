# Contributing to KPA

Thank you for considering contributing to `KPA`. It is much appreciated. Read this guide carefully to avoid unnecessary work and disappointment for anyone involved.

## Getting started

Set up the development environment

```
git clone git@github.com:VietHoang1512/KPA.git
cd KPA
pip install -r requirements.txt
pip install black isort flake8 pre-commit
pre-commit install
```

## Coding styles

Run `pre-commit install` to install pre-commit into your git hooks. pre-commit will now run on every commit. Otherwise you could manually run all pre-commit hooks for all files:
``` 
pre-commit run --all-files 
```