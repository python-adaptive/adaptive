import math

import pytest
import numpy as np
from scipy.spatial import ConvexHull

from adaptive.learner import LearnerND
from adaptive.learner.learner1D import with_pandas
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


@pytest.mark.skipif(not with_pandas, reason="pandas is not installed")
def test_learnerND_resume_after_loading_dataframe_convex_hull(monkeypatch):
    import pandas
    from types import MethodType

    hull_points = [
        (4.375872112626925, 8.917730007820797),
        (4.236547993389047, 6.458941130666561),
        (6.027633760716439, 5.448831829968968),
        (9.636627605010293, 3.8344151882577773),
    ]

    data_points = [
        (4.375872112626925, 8.917730007820797),
        (4.236547993389047, 6.458941130666561),
        (6.027633760716439, 5.448831829968968),
        (9.636627605086398, 3.834415188269945),
        (0.7103605819788694, 0.8712929970154071),
        (0.2021839744032572, 8.32619845547938),
        (7.781567509498505, 8.700121482468191),
    ]

    df = pandas.DataFrame(data_points, columns=["x", "y"])
    df["value"] = df["x"] + df["y"]

    hull = ConvexHull(hull_points)

    def some_f(xy):
        return xy[0] + xy[1]

    learner_old = LearnerND(some_f, hull)
    learner_old.load_dataframe(
        df,
        with_default_function_args=False,
        point_names=("x", "y"),
        value_name="value",
    )

    def old_ask_bound_point(self):
        new_point = next(
            p for p in self._bounds_points if p not in self.data and p not in self.pending_points
        )
        self.tell_pending(new_point)
        return new_point, np.inf

    learner_old._ask_bound_point = MethodType(old_ask_bound_point, learner_old)

    def naive_is_known_point(self, point):
        point = tuple(map(float, point))
        return point in self.data or point in self.pending_points

    learner_old._is_known_point = MethodType(naive_is_known_point, learner_old)
    learner_old._bound_match_tol = 0.0

    with pytest.raises(ValueError):
        BlockingRunner(learner_old, npoints_goal=len(df) + 1)

    learner = LearnerND(some_f, hull)
    learner.load_dataframe(
        df,
        with_default_function_args=False,
        point_names=("x", "y"),
        value_name="value",
    )

    target = len(df) + 1
    BlockingRunner(learner, npoints_goal=target)
    assert learner.npoints >= target
