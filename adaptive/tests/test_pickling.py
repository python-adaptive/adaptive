import pickle

import pytest

from adaptive.learner import (
    AverageLearner,
    BalancingLearner,
    DataSaver,
    IntegratorLearner,
    Learner1D,
    Learner2D,
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


def pickleable_f(x):
    return hash(str(x)) / 2 ** 63


nonpickleable_f = lambda x: hash(str(x)) / 2 ** 63  # noqa: E731


def identity_function(x):
    return x


def datasaver(f, learner_type, learner_kwargs):
    return DataSaver(
        learner=learner_type(f, **learner_kwargs), arg_picker=identity_function
    )


def balancing_learner(f, learner_type, learner_kwargs):
    learner_1 = learner_type(f, **learner_kwargs)
    learner_2 = learner_type(f, **learner_kwargs)
    return BalancingLearner([learner_1, learner_2])


learners_pairs = [
    (Learner1D, dict(bounds=(-1, 1))),
    (Learner2D, dict(bounds=[(-1, 1), (-1, 1)])),
    (SequenceLearner, dict(sequence=list(range(100)))),
    (IntegratorLearner, dict(bounds=(0, 1), tol=1e-3)),
    (AverageLearner, dict(atol=0.1)),
    (datasaver, dict(learner_type=Learner1D, learner_kwargs=dict(bounds=(-1, 1)))),
    (
        balancing_learner,
        dict(learner_type=Learner1D, learner_kwargs=dict(bounds=(-1, 1))),
    ),
]

serializers = [(pickle, pickleable_f)]
if with_cloudpickle:
    serializers.append((cloudpickle, nonpickleable_f))
if with_dill:
    serializers.append((dill, nonpickleable_f))


learners = [
    (learner_type, learner_kwargs, serializer, f)
    for serializer, f in serializers
    for learner_type, learner_kwargs in learners_pairs
]


@pytest.mark.parametrize("learner_type, learner_kwargs, serializer, f", learners)
def test_serialization_for(learner_type, learner_kwargs, serializer, f):
    """Test serializing a learner using different serializers."""

    learner = learner_type(f, **learner_kwargs)

    simple(learner, goal_1)
    learner_bytes = serializer.dumps(learner)
    loss = learner.loss()
    asked = learner.ask(10)
    data = learner.data

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
