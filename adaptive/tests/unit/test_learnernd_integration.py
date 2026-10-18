import math

import pytest

from adaptive.learner import LearnerND
from adaptive.learner.learnerND import curvature_loss_function
from adaptive.runner import BlockingRunner
from adaptive.runner import simple as SimpleRunner


def ring_of_fire(xy, d=0.75):
    a = 0.2
    x, y = xy
    return x + math.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


def wave_1d(x):
    return math.sin(x[0] * 5)


ONE_D_BOUNDS = [(-1, 1)]
TWO_D_BOUNDS = [(-1, 1), (-1, 1)]


@pytest.mark.parametrize(
    ("function", "bounds"),
    [(ring_of_fire, TWO_D_BOUNDS), (wave_1d, ONE_D_BOUNDS)],
    ids=["2d", "1d"],
)
def test_learnerND_runs_to_10_points(function, bounds):
    learner = LearnerND(function, bounds=bounds)
    SimpleRunner(learner, npoints_goal=10)
    assert learner.npoints == 10


@pytest.mark.parametrize(
    ("function", "bounds"),
    [(ring_of_fire, TWO_D_BOUNDS), (wave_1d, ONE_D_BOUNDS)],
    ids=["2d", "1d"],
)
@pytest.mark.parametrize("execution_number", range(5))
def test_learnerND_runs_to_10_points_Blocking(function, bounds, execution_number):
    learner = LearnerND(function, bounds=bounds)
    BlockingRunner(learner, npoints_goal=10)
    assert learner.npoints >= 10


@pytest.mark.parametrize(
    ("function", "bounds"),
    [(ring_of_fire, TWO_D_BOUNDS), (wave_1d, ONE_D_BOUNDS)],
    ids=["2d", "1d"],
)
def test_learnerND_curvature_runs_to_10_points(function, bounds):
    loss = curvature_loss_function()
    learner = LearnerND(function, bounds=bounds, loss_per_simplex=loss)
    SimpleRunner(learner, npoints_goal=10)
    assert learner.npoints == 10


@pytest.mark.parametrize("execution_number", range(5))
def test_learnerND_curvature_runs_to_10_points_Blocking(execution_number):
    loss = curvature_loss_function()
    learner = LearnerND(ring_of_fire, bounds=TWO_D_BOUNDS, loss_per_simplex=loss)
    BlockingRunner(learner, npoints_goal=10)
    assert learner.npoints >= 10


def test_learnerND_log_works():
    loss = curvature_loss_function()
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
    learner.ask(4)
    learner.tell((-1, -1), -1.0)
    learner.ask(1)
    learner.tell((-1, 1), -1.0)
    learner.tell((1, -1), 1.0)
    learner.ask(2)
    # At this point, there should! be one simplex in the triangulation,
    # furthermore the last two points that were asked should be in this simplex


def test_learnerND_1d_loss_decreases():
    """Test that loss decreases as more points are added."""
    learner = LearnerND(wave_1d, bounds=ONE_D_BOUNDS)
    SimpleRunner(learner, npoints_goal=5)
    loss_5 = learner.loss()
    assert loss_5 != float("inf")
    SimpleRunner(learner, npoints_goal=20)
    loss_20 = learner.loss()
    assert loss_20 <= loss_5
