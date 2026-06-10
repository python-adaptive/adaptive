import math

import pytest
from scipy.spatial import ConvexHull

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
    SimpleRunner(learner, npoints_goal=10)
    assert learner.npoints == 10


@pytest.mark.parametrize("execution_number", range(5))
def test_learnerND_runs_to_10_points_Blocking(execution_number):
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)])
    BlockingRunner(learner, npoints_goal=10)
    assert learner.npoints >= 10


def test_learnerND_curvature_runs_to_10_points():
    loss = curvature_loss_function()
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
    SimpleRunner(learner, npoints_goal=10)
    assert learner.npoints == 10


@pytest.mark.parametrize("execution_number", range(5))
def test_learnerND_curvature_runs_to_10_points_Blocking(execution_number):
    loss = curvature_loss_function()
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
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


def test_learnerND_resume_after_loading_dataframe_convex_hull():
    # Regression test for https://github.com/python-adaptive/adaptive/issues/470
    pandas = pytest.importorskip("pandas")

    hull_points = [
        (4.375872112626925, 8.917730007820797),
        (4.236547993389047, 6.458941130666561),
        (6.027633760716439, 5.448831829968968),
        (9.636627605010293, 3.8344151882577773),
    ]

    # Simulate float drift from a dataframe round-trip: one hull vertex is
    # off by 1e-10, so exact membership checks miss it and the learner used
    # to re-ask it, crashing with "Point already in triangulation.".
    drifted = tuple(c + 1e-10 for c in hull_points[-1])
    data_points = [*hull_points[:-1], drifted, (7.0, 6.0)]

    df = pandas.DataFrame(data_points, columns=["x", "y"])
    df["value"] = df["x"] + df["y"]

    def some_f(xy):
        return xy[0] + xy[1]

    learner = LearnerND(some_f, ConvexHull(hull_points))
    learner.load_dataframe(
        df,
        with_default_function_args=False,
        point_names=("x", "y"),
        value_name="value",
    )

    target = len(df) + 1
    BlockingRunner(learner, npoints_goal=target)
    assert learner.npoints >= target


def test_learnerND_remove_unfinished_reasks_bound_points():
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)])
    points, _ = learner.ask(4)
    assert set(points) == set(learner._bounds_points)

    # Discarding the pending bound points must make them available again.
    learner.remove_unfinished()
    points, _ = learner.ask(4)
    assert set(points) == set(learner._bounds_points)
