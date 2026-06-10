"""Tests for the automatic triangulation backend selection."""

import os
import subprocess
import sys

import pytest

from adaptive.learner import triangulation as python_triangulation
from adaptive.learner import triangulation_backend as backend
from adaptive.learner.triangulation_backend import (
    _MIN_RUST_VERSION,
    _rust_version,
)


def _run(code, backend_env):
    env = {**os.environ, "ADAPTIVE_TRIANGULATION_BACKEND": backend_env}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def rust_is_usable():
    version = _rust_version()
    return version is not None and version >= _MIN_RUST_VERSION


def test_backend_matches_installation():
    if rust_is_usable():
        import adaptive_triangulation

        assert backend.TRIANGULATION_BACKEND == "rust"
        assert backend.Triangulation is adaptive_triangulation.Triangulation
    else:
        assert backend.TRIANGULATION_BACKEND == "python"
        assert backend.Triangulation is python_triangulation.Triangulation


def test_python_triangulation_is_never_shadowed():
    # Old pickles reference adaptive.learner.triangulation.Triangulation by
    # qualified name, so the pure-Python class must stay importable as itself.
    assert python_triangulation.Triangulation.__module__ == (
        "adaptive.learner.triangulation"
    )


def test_force_python_backend():
    code = (
        "from adaptive.learner import triangulation, triangulation_backend;"
        "assert triangulation_backend.TRIANGULATION_BACKEND == 'python';"
        "assert triangulation_backend.Triangulation is triangulation.Triangulation"
    )
    result = _run(code, "python")
    assert result.returncode == 0, result.stderr


def test_force_rust_backend():
    code = (
        "from adaptive.learner import triangulation_backend;"
        "assert triangulation_backend.TRIANGULATION_BACKEND == 'rust'"
    )
    result = _run(code, "rust")
    if rust_is_usable():
        assert result.returncode == 0, result.stderr
    else:
        # Forcing the Rust backend without (a recent enough version of)
        # adaptive-triangulation must raise a helpful ImportError.
        assert result.returncode != 0
        assert "ImportError" in result.stderr
        assert "adaptive-triangulation" in result.stderr


def test_invalid_backend_raises():
    result = _run("import adaptive.learner.triangulation_backend", "bogus")
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_resolve_triangulation_class():
    resolve = backend.resolve_triangulation_class
    assert resolve("auto") is backend.Triangulation
    assert resolve("python") is python_triangulation.Triangulation
    if rust_is_usable():
        import adaptive_triangulation

        assert resolve("rust") is adaptive_triangulation.Triangulation
    else:
        with pytest.raises(ImportError, match="adaptive-triangulation"):
            resolve("rust")

    class MyTriangulation(python_triangulation.Triangulation):
        pass

    assert resolve(MyTriangulation) is MyTriangulation
    with pytest.raises(ValueError, match="Invalid triangulation backend"):
        resolve("bogus")


def test_learnernd_triangulation_backend_argument():
    from adaptive import LearnerND

    learner = LearnerND(
        lambda xy: sum(xy) ** 2,
        bounds=[(-1, 1), (-1, 1)],
        triangulation_backend="python",
    )
    assert learner._triangulation_class is python_triangulation.Triangulation
    assert learner.new()._triangulation_class is python_triangulation.Triangulation


@pytest.mark.skipif(not rust_is_usable(), reason="needs adaptive-triangulation")
def test_learnernd_uses_rust_backend():
    import adaptive_triangulation

    from adaptive import LearnerND

    learner = LearnerND(lambda xy: sum(xy) ** 2, bounds=[(-1, 1), (-1, 1)])
    for _ in range(50):
        points, _ = learner.ask(1)
        for point in points:
            learner.tell(point, learner.function(point))
    assert isinstance(learner.tri, adaptive_triangulation.Triangulation)
    assert learner.npoints >= 50


def test_rust_default_loss_matches_backend():
    if backend.TRIANGULATION_BACKEND == "rust":
        import adaptive_triangulation

        assert backend.rust_default_loss is adaptive_triangulation.default_loss
    else:
        assert backend.rust_default_loss is None


def _ring_of_fire(xy):
    import numpy as np

    x, y = xy
    a, d = 0.2, 0.5
    return x + np.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


@pytest.mark.skipif(not rust_is_usable(), reason="needs adaptive-triangulation")
def test_rust_backend_samples_identical_points():
    # The batched tell_pending path and the Rust default loss must not change
    # which points the learner chooses.
    from adaptive import LearnerND

    learners = {
        which: LearnerND(
            _ring_of_fire, bounds=[(-1, 1), (-1, 1)], triangulation_backend=which
        )
        for which in ("python", "rust")
    }
    for learner in learners.values():
        for _ in range(200):
            points, _ = learner.ask(1)
            for point in points:
                learner.tell(point, learner.function(point))
    assert sorted(learners["python"].data) == sorted(learners["rust"].data)
