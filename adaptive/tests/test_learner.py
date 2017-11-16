# -*- coding: utf-8 -*-

import collections
import inspect
import itertools as it
import functools as ft
import random
import math
import numpy as np
import scipy.spatial

import pytest

from ..learner import *


def generate_random_parametrization(f):
    """Return a realization of 'f' with parameters bound to random values.

    Parameters
    ----------
    f : callable
        All parameters but the first must be annotated with a callable
        that, when called with no arguments, produces a value of the
        appropriate type for the parameter in question.
    """
    _, *params = inspect.signature(f).parameters.items()
    if any(not callable(v.annotation) for (p, v) in params):
        raise TypeError('All parameters to {} must be annotated with functions.'
                        .format(f.__name__))
    realization = {p: v.annotation() for (p, v) in params}
    return ft.partial(f, **realization)


def uniform(a, b):
    return lambda: random.uniform(a, b)


# Library of functions and associated learners.

learner_function_combos = collections.defaultdict(list)

def learn_with(learner_type, **init_kwargs):

    def _(f):
        learner_function_combos[learner_type].append((f, init_kwargs))
        return f

    return _


# All parameters except the first must be annotated with a callable that
# returns a random value for that parameter.


@learn_with(Learner1D, bounds=(-1, 1))
def linear(x, m: uniform(0, 10)):
    return m * x


@learn_with(Learner1D, bounds=(-1, 1))
def linear_with_peak(x, d: uniform(-1, 1)):
    a = 0.01
    return x + a**2 / (a**2 + (x - d)**2)


@learn_with(Learner2D, bounds=((-1, 1), (-1, 1)))
def ring_of_fire(xy, d: uniform(0.2, 1)):
    a = 0.2
    x, y = xy
    return x + math.exp(-(x**2 + y**2 - d**2)**2 / a**4)


@learn_with(AverageLearner, rtol=1)
def gaussian(n):
    return random.gauss(0, 1)


# Decorators for tests.


def run_with(*learner_types):
    return pytest.mark.parametrize(
        'learner_type, f, learner_kwargs',
        [(l, f, k)
         for l in learner_types
         for f, k in learner_function_combos[l]]
    )


def choose_points_randomly(learner, rounds, points):
    n_rounds = random.randrange(*rounds)
    n_points = [random.randrange(*points) for _ in range(n_rounds)]

    xs = []
    ls = []
    for n in n_points:
        x, l = learner.choose_points(n)
        xs.extend(x)
        ls.extend(l)

    return xs, ls


@run_with(Learner1D)
def test_uniform_sampling1D(learner_type, f, learner_kwargs):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)

    points, _ = choose_points_randomly(learner, (10, 20), (10, 20))

    points.sort()
    ivals = np.diff(sorted(points))
    assert max(ivals) / min(ivals) < 2 + 1e-8


@pytest.mark.xfail
@run_with(Learner2D)
def test_uniform_sampling2D(learner_type, f, learner_kwargs):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)

    points, _ = choose_points_randomly(learner, (70, 100), (10, 20))
    tree = scipy.spatial.cKDTree(points)

    # regular grid
    n = math.sqrt(len(points))
    xbounds, ybounds = learner_kwargs['bounds']
    r = math.sqrt((ybounds[1] - ybounds[0]) / (xbounds[1] - xbounds[0]))
    xs, dx = np.linspace(*xbounds, int(n / r), retstep=True)
    ys, dy = np.linspace(*ybounds, int(n * r), retstep=True)

    distances, neighbors = tree.query(list(it.product(xs, ys)), k=1)
    assert max(distances) < math.sqrt(dx**2 + dy**2)


@run_with(Learner1D, Learner2D)
def test_adding_existing_data_is_idempotent(learner_type, f, learner_kwargs):
    """Adding already existing data is an idempotent operation.

    Either it is idempotent, or it is an error.
    This is the only sane behaviour.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner_type(f, **learner_kwargs)

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


@run_with(Learner1D, Learner2D, AverageLearner)
def test_adding_non_chosen_data(learner_type, f, learner_kwargs):
    """Adding data for a point that was not returned by 'choose_points'."""
    # XXX: learner, control and bounds are not defined
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner_type(f, **learner_kwargs)

    N = random.randint(10, 30)
    xs, _ = control.choose_points(N)

    for x in xs:
        control.add_point(x, f(x))
        learner.add_point(x, f(x))

    M = random.randint(10, 30)
    pls = zip(*learner.choose_points(M))
    cpls = zip(*control.choose_points(M))
    # Point ordering within a single call to 'choose_points'
    # is not guaranteed to be the same by the API.
    assert set(pls) == set(cpls)


@run_with(Learner1D, Learner2D, AverageLearner)
def test_point_adding_order_is_irrelevant(learner_type, f, learner_kwargs):
    """The order of calls to 'add_points' between calls to
       'choose_points' is arbitrary."""
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner_type(f, **learner_kwargs)

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
    # Point ordering within a single call to 'choose_points'
    # is not guaranteed to be the same by the API.
    assert set(pls) == set(cpls)


@run_with(Learner1D, Learner2D, AverageLearner)
def test_expected_loss_improvement_is_less_than_total_loss(learner_type, f, learner_kwargs):
    """The estimated loss improvement can never be greater than the total loss."""
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    N = random.randint(50, 100)
    xs, loss_improvements = learner.choose_points(N)

    for x in xs:
        learner.add_point(x, f(x))

    M = random.randint(50, 100)
    _, loss_improvements = learner.choose_points(M)

    assert sum(loss_improvements) < learner.loss()


@run_with(Learner1D, Learner2D)
def test_learner_subdomain(learner_type, f, learner_kwargs):
    """Learners that never receive data outside of a subdomain should
       perform 'similarly' to learners defined on that subdomain only."""
    # XXX: need the concept of a "subdomain"
    raise NotImplementedError()


@run_with(Learner1D, Learner2D)
def test_learner_performance_is_invariant_under_scaling(learner_type, f, learner_kwargs):
    """Learners behave identically under transformations that leave
       the loss invariant.

    This is a statement that the learner makes decisions based solely
    on the loss function.
    """
    # for now we just scale X and Y by random factors
    f = generate_random_parametrization(f)

    control_kwargs = dict(learner_kwargs)
    control = learner_type(f, **control_kwargs)

    xscale = 1000 * random.random()
    yscale = 1000 * random.random()

    l_kwargs = dict(learner_kwargs)
    l_kwargs['bounds'] = xscale * np.array(l_kwargs['bounds'])
    learner = learner_type(lambda x: yscale * f(x),
                           **l_kwargs)

    nrounds = random.randrange(50, 100)
    npoints = [random.randrange(1, 10) for _ in range(nrounds)]

    control_points = []
    for n in npoints:
        cxs, _ = control.choose_points(n)
        xs, _ = learner.choose_points(n)
        # Point ordering within a single call to 'choose_points'
        # is not guaranteed to be the same by the API.
        # Also, points will only be equal up to a tolerance, due to rounding
        should_be = sorted(cxs)
        to_check = np.array(sorted(xs)) / xscale
        assert np.allclose(should_be, to_check)

        control.add_data(cxs, [control.function(x) for x in cxs])
        learner.add_data(xs, [learner.function(x) for x in xs])


@run_with(Learner1D, Learner2D)
def test_convergence_for_arbitrary_ordering(learner_type, f, learner_kwargs):
    """Learners that are learning the same function should converge
    to the same result "eventually" if given the same data, regardless
    of the order in which that data is given.
    """
    raise NotImplementedError()
