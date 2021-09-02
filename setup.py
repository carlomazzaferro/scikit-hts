#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()


with open("HISTORY.rst") as history_file:
    history = history_file.read()

EXTENSIONS = {"auto_arima", "prophet", "geo", "test", "dev", "distributed", "all"}


def strip_comments(l):
    return l.split("#", 1)[0].strip()


def _pip_requirement(req, *root):
    if req.startswith("-r "):
        _, path = req.split()
        return reqs(*root, *path.split("/"))
    return [req]


def _reqs(*f):
    path = (Path.cwd() / "reqs").joinpath(*f)
    with path.open() as fh:
        reqs = [strip_comments(l) for l in fh.readlines()]
        return [_pip_requirement(r, *f[:-1]) for r in reqs if r]


def reqs(*f):
    return [req for subreq in _reqs(*f) for req in subreq]


def extras(*p):
    """Parse requirement in the requirements/extras/ directory."""
    return reqs("extras", *p)


def extras_require():
    """Get map of all extra requirements."""
    return {x: extras(x + ".txt") for x in EXTENSIONS}


install_requires = reqs("base.txt")
test_requires = (
    extras("prophet.txt")
    + extras("auto_arima.txt")
    + extras("distributed.txt")
    + extras("test.txt")
    + install_requires
)

setup(
    author="Carlo Mazzaferro",
    author_email="carlo.mazzaferro@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Hierarchical Time Series forecasting",
    install_requires=install_requires,
    extras_require=extras_require(),
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="scikit-hts",
    name="scikit-hts",
    packages=find_packages(include=["hts"]),
    test_suite="tests",
    tests_require=test_requires,
    version="0.5.12",
    zip_safe=False,
)
