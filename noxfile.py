"""Nox configuration file."""

import nox

nox.options.default_venv_backend = "uv"

python = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(python=python)
def pytest_min_deps(session: nox.Session) -> None:
    """Run pytest with no optional dependencies."""
    session.install(".[test]")
    session.run("coverage", "erase")
    session.run("pytest")


@nox.session(python=python)
def pytest_all_deps(session: nox.Session) -> None:
    """Run pytest with "other" optional dependencies."""
    session.install(".[test,other]")
    session.run("coverage", "erase")
    session.run("pytest")


@nox.session(python="3.13")
def pytest_typeguard(session: nox.Session) -> None:
    """Run pytest with typeguard."""
    session.install(".[test,other]")
    session.run("coverage", "erase")
    session.run("pytest", "--typeguard-packages=adaptive")


@nox.session(python="3.13")
def coverage(session: nox.Session) -> None:
    """Generate coverage report."""
    session.install(".[test,other]")
    session.run("pytest")

    session.run("coverage", "report")
    session.run("coverage", "xml")
