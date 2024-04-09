import nox


@nox.session(python=["3.9", "3.10", "3.11"])
@nox.parametrize("all_deps", [True, False])
def pytest(session, all_deps) -> None:
    session.install(".[testing,other]" if all_deps else ".[testing]")
    session.run("coverage", "erase")
    session.run("pytest")


@nox.session(python="3.11")
def pytest_typeguard(session) -> None:
    session.install(".[testing,other]")
    session.run("coverage", "erase")
    session.run("pytest", "--typeguard-packages=adaptive")


@nox.session(python="3.11")
def coverage(session) -> None:
    session.install("coverage")
    session.install(".[testing,other]")
    session.run("pytest")

    session.run("coverage", "report")
    session.run("coverage", "xml")
