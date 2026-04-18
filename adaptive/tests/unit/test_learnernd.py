import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

import adaptive.notebook_integration as notebook_integration
from adaptive.learner.base_learner import uses_nth_neighbors
from adaptive.learner.learnerND import (
    LearnerND,
    curvature_loss_function,
    default_loss,
    std_loss,
    uniform_loss,
)


def ring_of_fire(xy):
    a = 0.2
    d = 0.7
    x, y = xy
    return x + math.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


def test_learnerND_inits_loss_depends_on_neighbors_correctly():
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)])
    assert learner.nth_neighbors == 0


def test_learnerND_curvature_inits_loss_depends_on_neighbors_correctly():
    loss = curvature_loss_function()
    assert loss.nth_neighbors == 1
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
    assert learner.nth_neighbors == 1


def test_learnerND_accepts_ConvexHull_as_input():
    triangle = ConvexHull([(0, 1), (2, 0), (0, 0)])
    learner = LearnerND(ring_of_fire, bounds=triangle)
    assert learner.nth_neighbors == 0
    assert np.allclose(learner._bbox, [(0, 2), (0, 1)])


def test_learnerND_raises_if_too_many_neigbors():
    @uses_nth_neighbors(2)
    def loss(*args):
        return 0

    assert loss.nth_neighbors == 2
    with pytest.raises(NotImplementedError):
        LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)


# ---- 1D-specific LearnerND tests ----


ONE_D_BOUNDS = [(-1, 1)]
ONE_D_POINTS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def f_1d(x):
    """Simple 1D test function."""
    return x[0] ** 2


def make_1d_learner(function=f_1d, **kwargs):
    return LearnerND(function, bounds=ONE_D_BOUNDS, **kwargs)


def tell_1d_points(learner, function=None, points=ONE_D_POINTS):
    function = learner.function if function is None else function
    for x in points:
        learner.tell((x,), function((x,)))


def initialize_1d_learner(**kwargs):
    learner = make_1d_learner(**kwargs)
    points, _ = learner.ask(2)
    for point in points:
        learner.tell(point, learner.function(point))
    return learner


def test_learnerND_1d_construction():
    """Test that LearnerND can be constructed with 1D bounds."""
    learner = make_1d_learner()
    assert learner.ndim == 1
    assert learner._bounds_points == [(-1,), (1,)]
    assert learner._bbox == ((-1.0, 1.0),)


@pytest.mark.parametrize(
    ("loss_fn", "expected_nth_neighbors"),
    [
        (None, 0),
        (curvature_loss_function(), 1),
    ],
    ids=["default", "curvature"],
)
def test_learnerND_1d_tell_ask(loss_fn, expected_nth_neighbors):
    """Test basic tell/ask cycle for 1D LearnerND."""
    kwargs = {} if loss_fn is None else {"loss_per_simplex": loss_fn}
    learner = initialize_1d_learner(**kwargs)

    assert learner.tri is not None
    assert learner.nth_neighbors == expected_nth_neighbors

    points2, losses2 = learner.ask(3)

    assert len(points2) == 3
    assert all(loss > 0 for loss in losses2)


@pytest.mark.parametrize(
    "loss_fn",
    [
        pytest.param(loss_fn, id=loss_fn.__name__)
        for loss_fn in (default_loss, uniform_loss, std_loss)
    ],
)
def test_learnerND_1d_loss_functions(loss_fn):
    """Test that all standard loss functions work for 1D."""
    learner = initialize_1d_learner(loss_per_simplex=loss_fn)
    points2, losses2 = learner.ask(3)

    assert len(points2) == 3
    assert all(loss > 0 for loss in losses2)


def test_learnerND_1d_interpolation():
    """Test that 1D interpolation works correctly."""
    learner = make_1d_learner()
    tell_1d_points(learner)
    ip = learner._ip()

    assert np.isclose(ip(0.0), 0.0)
    assert np.isclose(ip(1.0), 1.0)
    assert np.isclose(ip(0.25), 0.125)  # linear between 0 and 0.5


def test_learnerND_1d_vector_output_interpolation():
    """Test that 1D interpolation works for R^1 -> R^M functions."""

    def f_vec(x):
        return np.array([x[0] ** 2, np.sin(x[0])])

    learner = make_1d_learner(function=f_vec)
    tell_1d_points(learner, function=f_vec)
    ip = learner._ip()
    result = ip(0.0)
    assert result.shape == (2,)
    assert np.isclose(result[0], 0.0)
    assert np.isclose(result[1], 0.0)


def test_learnerND_1d_plot_requires_holoviews(monkeypatch):
    """Test that plotting fails with a clear error without holoviews."""

    import_module = notebook_integration.importlib.import_module

    def missing_holoviews(name):
        if name == "holoviews":
            raise ModuleNotFoundError
        return import_module(name)

    monkeypatch.setattr(
        notebook_integration.importlib, "import_module", missing_holoviews
    )

    learner = make_1d_learner()
    tell_1d_points(learner)

    with pytest.raises(
        RuntimeError, match="holoviews is not installed; plotting is disabled."
    ):
        learner.plot()


def test_learnerND_1d_plot():
    """Test that 1D plot() does not crash."""
    hv = pytest.importorskip("holoviews")

    hv.extension("bokeh")
    learner = make_1d_learner()
    tell_1d_points(learner)
    plot = learner.plot()
    assert plot is not None
