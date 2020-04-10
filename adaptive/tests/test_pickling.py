import random

import cloudpickle
import pytest

from adaptive.learner import (
    AverageLearner,
    BalancingLearner,
    DataSaver,
    IntegratorLearner,
    Learner1D,
    Learner2D,
    LearnerND,
    SequenceLearner,
)
from adaptive.runner import simple


def goal_1(learner):
    return learner.npoints >= 10


def goal_2(learner):
    return learner.npoints >= 20


@pytest.mark.parametrize(
    "learner_type, learner_kwargs",
    [
        (Learner1D, dict(bounds=(-1, 1))),
        (Learner2D, dict(bounds=[(-1, 1), (-1, 1)])),
        (LearnerND, dict(bounds=[(-1, 1), (-1, 1), (-1, 1)])),
        (SequenceLearner, dict(sequence=list(range(100)))),
        (IntegratorLearner, dict(bounds=(0, 1), tol=1e-3)),
        (AverageLearner, dict(atol=0.1)),
    ],
)
def test_cloudpickle_for(learner_type, learner_kwargs):
    """Test serializing a learner using cloudpickle.

    We use cloudpickle because with pickle the functions are only
    pickled by reference."""

    def f(x):
        return random.random()

    learner = learner_type(f, **learner_kwargs)

    simple(learner, goal_1)
    learner_bytes = cloudpickle.dumps(learner)

    # Delete references
    del f
    del learner

    learner_loaded = cloudpickle.loads(learner_bytes)
    assert learner_loaded.npoints >= 10
    simple(learner_loaded, goal_2)
    assert learner_loaded.npoints >= 20


def test_cloudpickle_for_datasaver():
    def f(x):
        return dict(x=1, y=x ** 2)

    _learner = Learner1D(f, bounds=(-1, 1))
    learner = DataSaver(_learner, arg_picker=lambda x: x["y"])

    simple(learner, goal_1)
    learner_bytes = cloudpickle.dumps(learner)

    # Delete references
    del f
    del _learner
    del learner

    learner_loaded = cloudpickle.loads(learner_bytes)
    assert learner_loaded.npoints >= 10
    simple(learner_loaded, goal_2)
    assert learner_loaded.npoints >= 20


def test_cloudpickle_for_balancing_learner():
    def f(x):
        return x ** 2

    learner_1 = Learner1D(f, bounds=(-1, 1))
    learner_2 = Learner1D(f, bounds=(-2, 2))
    learner = BalancingLearner([learner_1, learner_2])

    simple(learner, goal_1)
    learner_bytes = cloudpickle.dumps(learner)

    # Delete references
    del f
    del learner_1
    del learner_2
    del learner

    learner_loaded = cloudpickle.loads(learner_bytes)
    assert learner_loaded.npoints >= 10
    simple(learner_loaded, goal_2)
    assert learner_loaded.npoints >= 20
