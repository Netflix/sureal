# Contributing to SUREAL

If you would like to contribute code you can do so through GitHub by forking the repository and sending a pull request. When submitting code, please make every effort to follow existing conventions and style in order to keep the code as readable as possible.

## License

By contributing your code, you agree to license your contribution under the terms of the [Apache License v2.0](http://www.apache.org/licenses/LICENSE-2.0). Your contributions should also include the following header:

```
/**
 * Copyright 2018 the original author or authors.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

# Developer Guide

## Build and test

SUREAL uses [tox](https://pypi.org/project/tox/) to manage continuous integration with `Travis CI` on Github. tox supports automatic testing of multiple Python versions (currently 2.7 and 3.7). To use tox:

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

After code development:

- Verify that tests pass locally, run `tox`
- Add an entry to the `CHANGELOG.md`
- Commit your changes, merge your PR (with travis job succeeding)
- Once the commits are on `master`, apply a version tag by either:
    - running `python setup.py version --bump patch --push` (add `--commit` if output looks as expected; use `minor`/`major` instead of `patch` for minor or major version bump)
    - or editing `sureal/__init__.py` and perform a manual `git tag` (see example below)
- upload to PyPI with: `python setup.py sdist bdist_wheel upload`
- Verify latest version at: https://pypi.org/project/sureal/

Example version bump run:

```bash
~/dev/sureal: python setup.py version --bump patch --push

Not committing bump, use --commit to commit
Not pushing bump, use --push to push
Would update sureal/__init__.py:3 with: __version__ = "0.1.1"
Would run: git add sureal/__init__.py
Would run: git commit -m "Version 0.1.1"
Would run: git tag -a v0.1.1 -m "Version 0.1.1"
Would run: git push --tags origin
```
