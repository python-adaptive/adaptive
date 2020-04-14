import operator
import pickle
import random

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

try:
    import cloudpickle

    with_cloudpickle = True
except ModuleNotFoundError:
    with_cloudpickle = False

try:
    import dill

    with_dill = True
except ModuleNotFoundError:
    with_dill = False


def goal_1(learner):
    return learner.npoints == 10


def goal_2(learner):
    return learner.npoints == 20


learners_pairs = [
    (Learner1D, dict(bounds=(-1, 1))),
    (Learner2D, dict(bounds=[(-1, 1), (-1, 1)])),
    (LearnerND, dict(bounds=[(-1, 1), (-1, 1), (-1, 1)])),
    (SequenceLearner, dict(sequence=list(range(100)))),
    (IntegratorLearner, dict(bounds=(0, 1), tol=1e-3)),
    (AverageLearner, dict(atol=0.1)),
]

serializers = [pickle]
if with_cloudpickle:
    serializers.append(cloudpickle)
if with_dill:
    serializers.append(dill)

learners = [
    (learner_type, learner_kwargs, serializer)
    for serializer in serializers
    for learner_type, learner_kwargs in learners_pairs
]


def f_for_pickle(x):
    return 1


def f_for_pickle_datasaver(x):
    return dict(x=x, y=x)


@pytest.mark.parametrize(
    "learner_type, learner_kwargs, serializer", learners,
)
def test_serialization_for(learner_type, learner_kwargs, serializer):
    """Test serializing a learner using different serializers."""

    def f(x):
        return random.random()

    if serializer is pickle:
        # f from the local scope cannot be pickled
        f = f_for_pickle  # noqa: F811

    learner = learner_type(f, **learner_kwargs)

    simple(learner, goal_1)
    learner_bytes = serializer.dumps(learner)
    loss = learner.loss()
    asked = learner.ask(10)
    data = learner.data

    if serializer is not pickle:
        # With pickle the functions are only pickled by reference
        del f
        del learner

    learner_loaded = serializer.loads(learner_bytes)
    assert learner_loaded.npoints == 10
    assert loss == learner_loaded.loss()
    assert data == learner_loaded.data

    assert asked == learner_loaded.ask(10)

    # load again to undo the ask
    learner_loaded = serializer.loads(learner_bytes)

    simple(learner_loaded, goal_2)
    assert learner_loaded.npoints == 20


@pytest.mark.parametrize(
    "serializer", serializers,
)
def test_serialization_for_datasaver(serializer):
    def f(x):
        return dict(x=1, y=x ** 2)

    if serializer is pickle:
        # f from the local scope cannot be pickled
        f = f_for_pickle_datasaver  # noqa: F811

    _learner = Learner1D(f, bounds=(-1, 1))
    learner = DataSaver(_learner, arg_picker=operator.itemgetter("y"))

    simple(learner, goal_1)
    learner_bytes = serializer.dumps(learner)

    if serializer is not pickle:
        # With pickle the functions are only pickled by reference
        del f
        del _learner
        del learner

    learner_loaded = serializer.loads(learner_bytes)
    assert learner_loaded.npoints == 10
    simple(learner_loaded, goal_2)
    assert learner_loaded.npoints == 20


@pytest.mark.parametrize(
    "serializer", serializers,
)
def test_serialization_for_balancing_learner(serializer):
    def f(x):
        return x ** 2

    if serializer is pickle:
        # f from the local scope cannot be pickled
        f = f_for_pickle  # noqa: F811

    learner_1 = Learner1D(f, bounds=(-1, 1))
    learner_2 = Learner1D(f, bounds=(-2, 2))
    learner = BalancingLearner([learner_1, learner_2])

    simple(learner, goal_1)
    learner_bytes = serializer.dumps(learner)

    if serializer is not pickle:
        # With pickle the functions are only pickled by reference
        del f
        del learner_1
        del learner_2
        del learner

    learner_loaded = serializer.loads(learner_bytes)
    assert learner_loaded.npoints == 10
    simple(learner_loaded, goal_2)
    assert learner_loaded.npoints == 20
