import nox


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
@nox.parametrize("all_deps", [True, False])
def pytest(session, all_deps):
    if all_deps:
        session.install(".[testing,other]")
    else:
        session.install(".[testing]")
    session.install(".")
    session.run("coverage", "erase")
    session.run("pytest")


@nox.session(python="3.7")
def coverage(session):
    session.install("coverage")
    session.run("coverage", "report")
    session.run("coverage", "xml")

    session.install(".[testing,other]")
    session.install(".")
    session.run("pytest")
