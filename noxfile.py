import nox


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
@nox.parametrize("all_deps", [True, False])
def pytest(session, all_deps):
    session.install(".[testing,other]" if all_deps else ".[testing]")
    session.run("coverage", "erase")
    session.run("pytest")


@nox.session(python="3.10")
def pytest_typeguard(session):
    session.install(".[testing,other]")
    session.run("coverage", "erase")
    session.run("pytest", "--typeguard-packages=adaptive")


@nox.session(python="3.7")
def coverage(session):
    session.install("coverage")
    session.install(".[testing,other]")
    session.run("pytest")

    session.run("coverage", "report")
    session.run("coverage", "xml")
