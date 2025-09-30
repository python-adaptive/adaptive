import json
import math
from pathlib import Path

import numpy as np
import pytest
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
def test_learnerND_resume_after_loading_dataframe_convex_hull():
    import pandas

    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pandas.read_csv(data_dir / "issue_470_sampled_points.csv", sep=";")
    boundaries = json.loads((data_dir / "issue_470_boundaries.json").read_text())
    hull = ConvexHull(boundaries)

    def some_f(xy):
        x, y = xy
        a = 0.2
        return x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)

    learner = LearnerND(some_f, hull)
    learner.load_dataframe(
        df,
        with_default_function_args=False,
        point_names=("x", "y"),
    )

    target = len(df) + 10
    BlockingRunner(learner, npoints_goal=target)
    assert learner.npoints >= target
