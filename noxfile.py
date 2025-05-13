"""Nox configuration file."""

import os

import nox

nox.options.default_venv_backend = "uv"

python = ["3.9", "3.10", "3.11", "3.12", "3.13"]
num_cpus = os.cpu_count() or 1
xdist = ("-n", "auto") if num_cpus > 2 else ()


@nox.session(python=python)
def pytest_min_deps(session: nox.Session) -> None:
    """Run pytest with no optional dependencies."""
    session.install(".[test]")
    session.run("coverage", "erase")
    session.run("pytest", *xdist)


@nox.session(python=python)
def pytest_all_deps(session: nox.Session) -> None:
    """Run pytest with "other" optional dependencies."""
    session.install(".[test,other]")
    session.run("coverage", "erase")
    session.run("pytest", *xdist)


@nox.session(python="3.13")
def pytest_typeguard(session: nox.Session) -> None:
    """Run pytest with typeguard."""
    session.install(".[test,other]")
    session.run("coverage", "erase")
    session.run("pytest", "--typeguard-packages=adaptive", *xdist)


@nox.session(python="3.13")
def coverage(session: nox.Session) -> None:
    """Generate coverage report."""
    session.install(".[test,other]")
    session.run("pytest", *xdist)

    session.run("coverage", "report")
    session.run("coverage", "xml")
