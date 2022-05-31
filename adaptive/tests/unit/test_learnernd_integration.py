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


def test_learnerND_runs_to_10_points():
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)])
    SimpleRunner(learner, goal=lambda l: l.npoints >= 10)
    assert learner.npoints == 10


@pytest.mark.parametrize("execution_number", range(5))
def test_learnerND_runs_to_10_points_Blocking(execution_number):
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)])
    BlockingRunner(learner, goal=lambda l: l.npoints >= 10)
    assert learner.npoints >= 10


def test_learnerND_curvature_runs_to_10_points():
    loss = curvature_loss_function()
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
    SimpleRunner(learner, goal=lambda l: l.npoints >= 10)
    assert learner.npoints == 10


@pytest.mark.parametrize("execution_number", range(5))
def test_learnerND_curvature_runs_to_10_points_Blocking(execution_number):
    loss = curvature_loss_function()
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
    BlockingRunner(learner, goal=lambda l: l.npoints >= 10)
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
