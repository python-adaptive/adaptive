import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from adaptive.learner.base_learner import uses_nth_neighbors
from adaptive.learner.learnerND import LearnerND, curvature_loss_function


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


def f_1d(x):
    """Simple 1D test function."""
    return x[0] ** 2


def test_learnerND_1d_construction():
    """Test that LearnerND can be constructed with 1D bounds."""
    learner = LearnerND(f_1d, bounds=[(-1, 1)])
    assert learner.ndim == 1
    assert learner._bounds_points == [(-1,), (1,)]
    assert learner._bbox == ((-1.0, 1.0),)


def test_learnerND_1d_tell_ask():
    """Test basic tell/ask cycle for 1D LearnerND."""
    learner = LearnerND(f_1d, bounds=[(-1, 1)])
    # Ask for bound points first
    points, losses = learner.ask(2)
    assert len(points) == 2
    # Tell the boundary values
    for p in points:
        learner.tell(p, f_1d(p))
    # Now we should have a triangulation
    assert learner.tri is not None
    # Ask for more points
    points2, losses2 = learner.ask(3)
    assert len(points2) == 3


def test_learnerND_1d_loss_functions():
    """Test that all standard loss functions work for 1D."""
    from adaptive.learner.learnerND import (
        default_loss,
        std_loss,
        uniform_loss,
    )

    for loss_fn in [default_loss, uniform_loss, std_loss]:
        learner = LearnerND(f_1d, bounds=[(-1, 1)], loss_per_simplex=loss_fn)
        points, _ = learner.ask(2)
        for p in points:
            learner.tell(p, f_1d(p))
        points2, losses2 = learner.ask(3)
        assert len(points2) == 3
        assert all(l > 0 for l in losses2)


def test_learnerND_1d_curvature_loss():
    """Test that curvature loss function works for 1D."""
    loss = curvature_loss_function()
    learner = LearnerND(f_1d, bounds=[(-1, 1)], loss_per_simplex=loss)
    assert learner.nth_neighbors == 1
    points, _ = learner.ask(2)
    for p in points:
        learner.tell(p, f_1d(p))
    points2, _ = learner.ask(3)
    assert len(points2) == 3


def test_learnerND_1d_interpolation():
    """Test that 1D interpolation works correctly."""
    learner = LearnerND(f_1d, bounds=[(-1, 1)])
    # Tell some points
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        learner.tell((x,), x**2)
    ip = learner._ip()
    # Check interpolation at known points
    assert np.isclose(ip(0.0), 0.0)
    assert np.isclose(ip(1.0), 1.0)
    # Check interpolation at midpoint (linear interpolation)
    assert np.isclose(ip(0.25), 0.125)  # linear between 0 and 0.5


def test_learnerND_1d_vector_output_interpolation():
    """Test that 1D interpolation works for R^1 -> R^M functions."""

    def f_vec(x):
        return np.array([x[0] ** 2, np.sin(x[0])])

    learner = LearnerND(f_vec, bounds=[(-1, 1)])
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        learner.tell((x,), f_vec((x,)))
    ip = learner._ip()
    result = ip(0.0)
    assert result.shape == (2,)
    assert np.isclose(result[0], 0.0)
    assert np.isclose(result[1], 0.0)


def test_learnerND_1d_plot():
    """Test that 1D plot() does not crash."""
    import holoviews as hv

    hv.extension("bokeh")
    learner = LearnerND(f_1d, bounds=[(-1, 1)])
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        learner.tell((x,), x**2)
    plot = learner.plot()
    assert plot is not None
