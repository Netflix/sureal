# Developer Guide

To release a new version:

- bump the version and edit `sureal/version.py`
- add an entry to the `CHANGELOG.md`
- add a Git tag for the version with `git tag vXXX`
- upload to PyPI with `python setup.py sdist upload`