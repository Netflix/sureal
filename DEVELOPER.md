# Developer Guide

## Build and test

- Grab [tox](https://pypi.org/project/tox/)
- Create a venv:
```bash
tox -e venv
source .venv/bin/activate
which sureal
sureal
deactivate
```
- Run the tests:
```bash
# Run all the tests, with all python versions
tox

# Run tests with python 3.7 only
tox -e py37

# Quickly run just one given test:
tox -- -k test_btnr_subjective_model

# Same, but with python3.7 only:
tox -e py37 -- -k test_btnr_subjective_model

# Stop on first test failure:
tox -e py37 -- -x
```


## Release a new version

- Verify that tests pass locally, run `tox`
- Add an entry to the `CHANGELOG.md`
- Commit your changes, merge your PR (with travis job succeeding)
- Once the commits are on `master`, apply a version tag by either:
    - running `python setup.py version --bump patch` (add `--commit --push` if output looks as expected)
    - or edit `sureal/__init__.py` and perform a manual `git tag` (see example below)

- upload to PyPI with `python setup.py sdist upload` (this shouldn't be needed if we complete `.travis.yml`)


Example version bump run:

```bash
~/dev/sureal: python setup.py version --bump patch

Not committing bump, use --commit to commit
Not pushing bump, use --push to push
Would update sureal/__init__.py:3 with: __version__ = "0.1.1"
Would run: git add sureal/__init__.py
Would run: git commit -m "Version 0.1.1"
Would run: git tag -a v0.1.1 -m "Version 0.1.1"
```