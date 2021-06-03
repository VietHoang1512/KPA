# Contributing to KPA

Thank you for considering contributing to `KPA`. It is much appreciated. Read this guide carefully to avoid unnecessary work and disappointment for anyone involved.

## Getting started

Setting up your environment is pretty straight forward:

```
git clone git@github.com:VietHoang1512/KPA.git
pip install -r requirements.txt
pip install black isort flake8 pre-commit
pre-commit install
```

## Coding styles

We use pre-commit hooks (`black`, `isort`, `flake8`) to ensure the throughout coding style. The hooks will automatically run before you commiting your codes, you can also mannually run this with 
``` 
pre-commit run --all-files 
```