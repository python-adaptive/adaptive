# -*- coding: utf-8 -*-

import pytest

from ..learner import *

adaptive_learners = [
    Learner1D,
    Learner2D,
]

learners = [
    *adaptive_learners,
    AverageLearner,
]

adaptive_learners = pytest.mark.parametrize("learner_type", adaptive_learners)
learners = pytest.mark.parametrize("learner_type", learners)


@adaptive_learners
def test_uniform_sampling(learner_type):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    raise NotImplementedError()


@learners
def test_adding_existing_data_is_idempotent(learner_type):
    """Adding already existing data is an idempotent operation.

    Either it is idempotent, or it is an error.
    This is the only sane behaviour.
    """
    raise NotImplementedError()


@adaptive_learners
def test_adding_non_chosen_data(learner_type):
    """Adding data for a point that was not returned by 'choose_points'."""
    raise NotImplementedError()


@learners
def test_point_adding_order_is_irrelevant(learner_type):
    """The order of calls to 'add_points' between calls to
       'choose_points' is arbitrary."""
    raise NotImplementedError()


@learners
def test_expected_loss_improvement_is_less_than_total_loss(learner_type):
    """The estimated loss improvement can never be greater than the total loss."""
    raise NotImplementedError()


@adaptive_learners
def test_learner_subdomain(learner_type):
    """Learners that never receive data outside of a subdomain should
       perform 'similarly' to learners defined on that subdomain only."""
    raise NotImplementedError()


@learners
def test_learner_performance_is_invariant_under_scaling(learner_type):
    """Learners behave identically under transformations that leave
       the loss invariant.

    This is a statement that the learner makes decisions based solely
    on the loss function.
    """
    raise NotImplementedError()


@learners
def test_convergence_for_arbitrary_ordering(learner_type):
    """Learners that are learning the same function should converge
    to the same result "eventually" if given the same data, regardless
    of the order in which that data is given.
    """
    raise NotImplementedError()

@Learner2D
def test_choose_point_returns_tuple(learner):
    points, _ = learner.choose_points(1)
    for x in points:
        learner.add_point(x, learner.function(x))
        assert isinstance(x, tuple)
