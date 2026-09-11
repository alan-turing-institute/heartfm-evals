See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Setting up a development environment manually

This project uses [uv](https://astral.sh/uv) only — please don't install with
bare pip. Set up a development environment by running:

```zsh
uv venv .venv                # create a virtualenv called .venv
source .venv/bin/activate    # now `python` points to the virtualenv python
uv sync --all-extras         # install the project plus the `dev` extra
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
uv tool install pre-commit # or brew install pre-commit on macOS
pre-commit install # this will install a pre-commit hook into the git repo
```

`pre-commit install` is per-clone and is **not** done for you — without it none of
the ruff/mypy checks run at commit time, and CI will be the first thing to catch
them.

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=heartfm_evals
```

# Pre-commit

This project uses pre-commit for all style checking. Install pre-commit and run:

```bash
pre-commit run -a
```

to check all files.
