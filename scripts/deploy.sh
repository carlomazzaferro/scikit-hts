#!/usr/bin/env bash

set -euxo pipefail

python setup.py sdist bdist_wheel
pip install twine
twine upload --username Mazzafish --password $PYPI --skip-existing dist/*
