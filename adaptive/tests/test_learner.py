# -*- coding: utf-8 -*-

import collections
import inspect
import itertools as it
import functools as ft
import random
import math

import pytest

from ..learner import *


def get_annotations(f):
    """Return an ordered dict of parameter annotations for 'f'."""
    params = inspect.signature(f).parameters
    annot = collections.OrderedDict((p, params[p].annotation) for p in params)
    if any(a is inspect._empty for a in annot.values()):
        raise ValueError('function {} is missing annotated parameters'
                         .format(f.__name__))
    return annot


def generate_random_parametrization(f):
    """Return a realization of 'f' with parameters bound to random values.

    Parameters
    ----------
    f : callable
        All parameters but the first must be floats, and annotated
        with a pair (min, max); the bounds on the values of the parameter.
    """
    bounds = list(get_annotations(f).items())
    rparams = {name: a + (b - a) * random.random()
               for name, (a, b) in bounds[1:]}

    return ft.partial(f, **rparams)


# The annotation on the first parameter to 'f' are the bounds
def get_bounds(f):
    return next(iter(get_annotations(f)))


# Library of functions and learners that can learn them

learner_function_map = collections.defaultdict(list)

def learn_with(*learner_types):

    def _(f):
        get_annotations(f)  # Raise if function is not annotated
        for l in learner_types:
            learner_function_map[l].append(f)

    return _


@learn_with(Learner1D)
def linear(x:(-1, 1), m:(0, 10)):
    return m * x


@learn_with(Learner1D)
def linear_with_peak(x:(-1, 1), d:(-1, 1)):
    a = 0.01
    return x + a**2 / (a**2 + (x - d)**2)


@learn_with(Learner2D)
def ring_of_fire(xy:((-1, 1), (-1, 1)), d:(0.2, 1)):
    a = 0.2
    x, y = xy
    return x + math.exp(-(x**2 + y**2 - d**2)**2 / a**4)


@learn_with(AverageLearner)
def gaussian(n:dict(rtol=1)):
    return random.gauss(0, 1)


# Factories for building learners from a function.
# This encodes the convention for passing extra information
# (bounds for 1D and 2D, and rtol/atol for Averaging)
# in the function annotations. This is necessary because
# __init__ is not the same for all learners.

factories = {}

def factory(*learner_types):

    def _(builder):
        for l in learner_types:
            factories[l] = ft.partial(builder, l)
    return _


@factory(Learner1D, Learner2D)
def build(l, f):
    bounds = next(iter(get_annotations(f).values()))
    return l(f, bounds)


@factory(AverageLearner)
def build(l, f):
    kwargs = next(iter(get_annotations(f).values()))
    return l(f, **kwargs)


# Decorators for tests.

bounded_learners = [
    Learner1D,
    Learner2D,
]

learners = [
    *bounded_learners,
    AverageLearner,
]


# Learners with bounds
bounded_learners = pytest.mark.parametrize(
    "learner_type, f, bounds",
    [(l, f, get_bounds(f))
     for l in bounded_learners for f in learner_function_map[l]]
)

# In general the only property of a learner that we know about
# is the function 'f' that they are learning.
learners = pytest.mark.parametrize(
    "learner_factory,f",
    [(factories[l], f)
     for l in learners for f in learner_function_map[l]]
)


@pytest.mark.xfail
@bounded_learners
def test_uniform_sampling(learner_type, f, bounds):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    raise NotImplementedError()


@learners
def test_adding_existing_data_is_idempotent(learner_factory, f):
    """Adding already existing data is an idempotent operation.

    Either it is idempotent, or it is an error.
    This is the only sane behaviour.
    """
    f = generate_random_parametrization(f)
    learner = learner_factory(f)
    control = learner_factory(f)

    N = random.randint(10, 30)
    control.choose_points(N)
    xs, _ = learner.choose_points(N)
    points = [(x, f(x)) for x in xs]

    for p in points:
        control.add_point(*p)
        learner.add_point(*p)

    random.shuffle(points)
    for p in points:
        learner.add_point(*p)

    M = random.randint(10, 30)
    pls = zip(*learner.choose_points(M))
    cpls = zip(*control.choose_points(M))
    # Point ordering is not defined, so compare as sets
    assert set(pls) == set(cpls)


@learners
def test_adding_non_chosen_data(learner_factory, f):
    """Adding data for a point that was not returned by 'choose_points'."""
    # XXX: learner, control and bounds are not defined
    f = generate_random_parametrization(f)
    learner = learner_factory(f)
    control = learner_factory(f)

    N = random.randint(10, 30)
    xs, _ = control.choose_points(N)

    for x in xs:
        control.add_point(x, f(x))
        learner.add_point(x, f(x))

    M = random.randint(10, 30)
    pls = zip(*learner.choose_points(M))
    cpls = zip(*control.choose_points(M))
    # Point ordering is not defined, so compare as sets
    assert set(pls) == set(cpls)


@learners
def test_point_adding_order_is_irrelevant(learner_factory, f):
    """The order of calls to 'add_points' between calls to
       'choose_points' is arbitrary."""
    f = generate_random_parametrization(f)
    learner = learner_factory(f)
    control = learner_factory(f)

    N = random.randint(10, 30)
    control.choose_points(N)
    xs, _ = learner.choose_points(N)
    points = [(x, f(x)) for x in xs]

    for p in points:
        control.add_point(*p)

    random.shuffle(points)
    for p in points:
        learner.add_point(*p)

    M = random.randint(10, 30)
    pls = zip(*learner.choose_points(M))
    cpls = zip(*control.choose_points(M))
    # Point ordering is not defined, so compare as sets
    assert set(pls) == set(cpls)


@learners
def test_expected_loss_improvement_is_less_than_total_loss(learner_factory, f):
    """The estimated loss improvement can never be greater than the total loss."""
    f = generate_random_parametrization(f)
    learner = learner_factory(f)
    N = random.randint(50, 100)
    xs, loss_improvements = learner.choose_points(N)

    # no data -- loss is infinite
    assert all(l == float('inf') for l in loss_improvements)
    assert learner.loss() == float('inf')

    for x in xs:
        learner.add_point(x, f(x))

    M = random.randint(50, 100)
    _, loss_improvements = learner.choose_points(M)

    assert sum(loss_improvements) < learner.loss()


@pytest.mark.xfail
@bounded_learners
def test_learner_subdomain(learner_type, f, bounds):
    """Learners that never receive data outside of a subdomain should
       perform 'similarly' to learners defined on that subdomain only."""
    # XXX: need the concept of a "subdomain"
    raise NotImplementedError()


@pytest.mark.xfail
@bounded_learners
def test_learner_performance_is_invariant_under_scaling(learner_type, f, bounds):
    """Learners behave identically under transformations that leave
       the loss invariant.

    This is a statement that the learner makes decisions based solely
    on the loss function.
    """
    # XXX: neew the concept of "scaling"
    raise NotImplementedError()


@pytest.mark.xfail
@learners
def test_convergence_for_arbitrary_ordering(learner_factory, f):
    """Learners that are learning the same function should converge
    to the same result "eventually" if given the same data, regardless
    of the order in which that data is given.
    """
    raise NotImplementedError()
