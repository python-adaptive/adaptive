"""Nox configuration file."""

import nox


@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
@nox.parametrize("all_deps", [True, False])
def pytest(session: nox.Session, all_deps: bool) -> None:
    """Run pytest with optional dependencies."""
    session.install(".[testing,other]" if all_deps else ".[testing]")
    session.run("coverage", "erase")
    session.run("pytest")


@nox.session(python="3.11")
def pytest_typeguard(session: nox.Session) -> None:
    """Run pytest with typeguard."""
    session.install(".[testing,other]")
    session.run("coverage", "erase")
    session.run("pytest", "--typeguard-packages=adaptive")


@nox.session(python="3.11")
def coverage(session: nox.Session) -> None:
    """Generate coverage report."""
    session.install("coverage")
    session.install(".[testing,other]")
    session.run("pytest")

    session.run("coverage", "report")
    session.run("coverage", "xml")
