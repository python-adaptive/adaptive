# -*- coding: utf-8 -*-

import collections
import functools as ft
import inspect
import itertools as it
import math
import operator
import os
import random
import shutil
import tempfile

import numpy as np
import pytest
import scipy.spatial

from ..learner import (AverageLearner, BalancingLearner, DataSaver,
    IntegratorLearner, Learner1D, Learner2D, LearnerND)
from ..runner import simple


try:
    import skopt
    from ..learner import SKOptLearner
except ModuleNotFoundError:
    SKOptLearner = None


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


def xfail(learner):
    return pytest.mark.xfail, learner


def maybe_skip(learner):
    return (pytest.mark.skip, learner) if learner is None else learner


# All parameters except the first must be annotated with a callable that
# returns a random value for that parameter.


@learn_with(Learner1D, bounds=(-1, 1))
def quadratic(x, m: uniform(0, 10), b: uniform(0, 1)):
    return m * x**2 + b


@learn_with(Learner1D, bounds=(-1, 1))
def linear_with_peak(x, d: uniform(-1, 1)):
    a = 0.01
    return x + a**2 / (a**2 + (x - d)**2)


@learn_with(LearnerND, bounds=((-1, 1), (-1, 1)))
@learn_with(Learner2D, bounds=((-1, 1), (-1, 1)))
def ring_of_fire(xy, d: uniform(0.2, 1)):
    a = 0.2
    x, y = xy
    return x + math.exp(-(x**2 + y**2 - d**2)**2 / a**4)


@learn_with(LearnerND, bounds=((-1, 1), (-1, 1), (-1, 1)))
def sphere_of_fire(xyz, d: uniform(0.2, 1)):
    a = 0.2
    x, y, z = xyz
    return x + math.exp(-(x**2 + y**2 + z**2 - d**2)**2 / a**4) + z**2


@learn_with(AverageLearner, rtol=1)
def gaussian(n):
    return random.gauss(0, 1)


# Decorators for tests.

def run_with(*learner_types):
    pars = []
    for l in learner_types:
        has_marker = isinstance(l, tuple)
        if has_marker:
            marker, l = l
        for f, k in learner_function_combos[l]:
            # Check if learner was marked with our `xfail` decorator
            # XXX: doesn't work when feeding kwargs to xfail.
            if has_marker:
                pars.append(pytest.param(l, f, dict(k),
                                         marks=[marker]))
            else:
                pars.append((l, f, dict(k)))
    return pytest.mark.parametrize('learner_type, f, learner_kwargs', pars)


def ask_randomly(learner, rounds, points):
    n_rounds = random.randrange(*rounds)
    n_points = [random.randrange(*points) for _ in range(n_rounds)]

    xs = []
    ls = []
    for n in n_points:
        x, l = learner.ask(n)
        xs.extend(x)
        ls.extend(l)

    return xs, ls


# Tests

@run_with(Learner1D)
def test_uniform_sampling1D(learner_type, f, learner_kwargs):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)

    points, _ = ask_randomly(learner, (10, 20), (10, 20))

    points.sort()
    ivals = np.diff(sorted(points))
    assert max(ivals) / min(ivals) < 2 + 1e-8


@pytest.mark.xfail
@run_with(Learner2D, LearnerND)
def test_uniform_sampling2D(learner_type, f, learner_kwargs):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)

    points, _ = ask_randomly(learner, (70, 100), (10, 20))
    tree = scipy.spatial.cKDTree(points)

    # regular grid
    n = math.sqrt(len(points))
    xbounds, ybounds = learner_kwargs['bounds']
    r = math.sqrt((ybounds[1] - ybounds[0]) / (xbounds[1] - xbounds[0]))
    xs, dx = np.linspace(*xbounds, int(n / r), retstep=True)
    ys, dy = np.linspace(*ybounds, int(n * r), retstep=True)

    distances, neighbors = tree.query(list(it.product(xs, ys)), k=1)
    assert max(distances) < math.sqrt(dx**2 + dy**2)


@pytest.mark.parametrize('learner_type, bounds', [
    (Learner1D, (-1, 1)),
    (Learner2D, [(-1, 1), (-1, 1)]),
    (LearnerND, [(-1, 1), (-1, 1), (-1, 1)]),
])
def test_learner_accepts_lists(learner_type, bounds):
    def f(x):
        return [0, 1]

    learner = learner_type(f, bounds=bounds)
    simple(learner, goal=lambda l: l.npoints > 10)


@run_with(xfail(Learner1D), Learner2D, LearnerND)
def test_adding_existing_data_is_idempotent(learner_type, f, learner_kwargs):
    """Adding already existing data is an idempotent operation.

    Either it is idempotent, or it is an error.
    This is the only sane behaviour.

    This test will fail for the Learner1D because the losses are normalized by
    _scale which is updated after every point. After one iteration of adding
    points, the _scale could be different from what it was when calculating
    the losses of the intervals. Readding the points a second time means
    that the losses are now all normalized by the correct _scale.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner_type(f, **learner_kwargs)

    N = random.randint(10, 30)
    control.ask(N)
    xs, _ = learner.ask(N)
    points = [(x, f(x)) for x in xs]

    for p in points:
        control.tell(*p)
        learner.tell(*p)

    random.shuffle(points)
    for p in points:
        learner.tell(*p)

    M = random.randint(10, 30)
    pls = zip(*learner.ask(M))
    cpls = zip(*control.ask(M))
    # Point ordering is not defined, so compare as sets
    assert set(pls) == set(cpls)


# XXX: This *should* pass (https://gitlab.kwant-project.org/qt/adaptive/issues/84)
#      but we xfail it now, as Learner2D will be deprecated anyway
@run_with(Learner1D, xfail(Learner2D), LearnerND, AverageLearner)
def test_adding_non_chosen_data(learner_type, f, learner_kwargs):
    """Adding data for a point that was not returned by 'ask'."""
    # XXX: learner, control and bounds are not defined
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner_type(f, **learner_kwargs)

    if learner_type is Learner2D:
        # If the stack_size is bigger then the number of points added,
        # ask will return a point from the _stack.
        learner.stack_size = 1
        control.stack_size = 1

    N = random.randint(10, 30)
    xs, _ = control.ask(N)

    ys = [f(x) for x in xs]
    for x, y in zip(xs, ys):
        control.tell(x, y)
        learner.tell(x, y)

    M = random.randint(10, 30)
    pls = zip(*learner.ask(M))
    cpls = zip(*control.ask(M))
    # Point ordering within a single call to 'ask'
    # is not guaranteed to be the same by the API.
    assert set(pls) == set(cpls)


@run_with(xfail(Learner1D), xfail(Learner2D), xfail(LearnerND), AverageLearner)
def test_point_adding_order_is_irrelevant(learner_type, f, learner_kwargs):
    """The order of calls to 'tell' between calls to 'ask'
    is arbitrary.

    This test will fail for the Learner1D for the same reason as described in
    the doc-string in `test_adding_existing_data_is_idempotent`.

    This test will fail for the Learner2D because
    `interpolate.interpnd.estimate_gradients_2d_global` will give different
    outputs based on the order of the triangles and values in
    (ip.tri, ip.values). Therefore the _stack will contain different points.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner_type(f, **learner_kwargs)

    N = random.randint(10, 30)
    control.ask(N)
    xs, _ = learner.ask(N)
    points = [(x, f(x)) for x in xs]

    for p in points:
        control.tell(*p)

    random.shuffle(points)
    for p in points:
        learner.tell(*p)

    M = random.randint(10, 30)
    pls = zip(*learner.ask(M))
    cpls = zip(*control.ask(M))
    # Point ordering within a single call to 'ask'
    # is not guaranteed to be the same by the API.
    # We compare the sorted points instead of set, because the points
    # should only be identical up to machine precision.
    np.testing.assert_almost_equal(sorted(pls), sorted(cpls))


# XXX: the Learner2D fails with ~50% chance
# see https://gitlab.kwant-project.org/qt/adaptive/issues/84
@run_with(Learner1D, xfail(Learner2D), LearnerND, AverageLearner)
def test_expected_loss_improvement_is_less_than_total_loss(learner_type, f, learner_kwargs):
    """The estimated loss improvement can never be greater than the total loss."""
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    N = random.randint(50, 100)
    xs, loss_improvements = learner.ask(N)

    for x in xs:
        learner.tell(x, f(x))

    M = random.randint(50, 100)
    _, loss_improvements = learner.ask(M)

    if learner_type is Learner2D:
        assert (sum(loss_improvements)
                < sum(learner.loss_per_triangle(learner.ip())))
    elif learner_type is Learner1D:
        assert sum(loss_improvements) < sum(learner.losses.values())
    elif learner_type is AverageLearner:
        assert sum(loss_improvements) < learner.loss()


# XXX: This *should* pass (https://gitlab.kwant-project.org/qt/adaptive/issues/84)
#      but we xfail it now, as Learner2D will be deprecated anyway
@run_with(Learner1D, xfail(Learner2D), xfail(LearnerND))
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
    learner = learner_type(lambda x: yscale * f(np.array(x) / xscale),
                           **l_kwargs)

    npoints = random.randrange(1000, 2000)

    for n in range(npoints):
        cxs, _ = control.ask(1)
        xs, _ = learner.ask(1)
        control.tell_many(cxs, [control.function(x) for x in cxs])
        learner.tell_many(xs, [learner.function(x) for x in xs])

        # Check whether the points returned are the same
        xs_unscaled = np.array(xs) / xscale
        assert np.allclose(xs_unscaled, cxs)

    # Check if the losses are close
    assert abs(learner.loss() - control.loss()) / learner.loss() < 1e-11


@run_with(Learner1D, Learner2D, LearnerND, AverageLearner)
def test_balancing_learner(learner_type, f, learner_kwargs):
    """Test if the BalancingLearner works with the different types of learners."""
    learners = [learner_type(generate_random_parametrization(f), **learner_kwargs)
                for i in range(4)]

    learner = BalancingLearner(learners)

    # Emulate parallel execution
    stash = []

    for i in range(100):
        n = random.randint(1, 10)
        m = random.randint(0, n)
        xs, _ = learner.ask(n, tell_pending=False)

        # Save 'm' random points out of `xs` for later
        random.shuffle(xs)
        for _ in range(m):
            stash.append(xs.pop())

        for x in xs:
            learner.tell(x, learner.function(x))

        # Evaluate and add 'm' random points from `stash`
        random.shuffle(stash)
        for _ in range(m):
            x = stash.pop()
            learner.tell(x, learner.function(x))

    assert all(l.npoints > 10 for l in learner.learners), [l.npoints for l in learner.learners]


@run_with(Learner1D, Learner2D, LearnerND, AverageLearner,
    maybe_skip(SKOptLearner), IntegratorLearner)
def test_saving(learner_type, f, learner_kwargs):
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner_type(f, **learner_kwargs)
    simple(learner, lambda l: l.npoints > 100)
    fd, path = tempfile.mkstemp()
    try:
        learner.save(path)
        control.load(path)
        if learner_type is not Learner1D:
            # Because different scales result in differnt losses
            np.testing.assert_almost_equal(learner.loss(), control.loss())

        # Try if the control is runnable
        simple(control, lambda l: l.npoints > 200)
    finally:
        os.remove(path)


@run_with(Learner1D, Learner2D, LearnerND, AverageLearner,
    maybe_skip(SKOptLearner), IntegratorLearner)
def test_saving_of_balancing_learner(learner_type, f, learner_kwargs):
    f = generate_random_parametrization(f)
    learner = BalancingLearner([learner_type(f, **learner_kwargs)])
    control = BalancingLearner([learner_type(f, **learner_kwargs)])

    # set fnames
    learner.learners[0].fname = 'test'
    control.learners[0].fname = 'test'

    simple(learner, lambda l: l.learners[0].npoints > 100)
    folder = tempfile.mkdtemp()
    try:
        learner.save(folder=folder)
        control.load(folder=folder)
        if learner_type is not Learner1D:
            # Because different scales result in differnt losses
            np.testing.assert_almost_equal(learner.loss(), control.loss())

        # Try if the control is runnable
        simple(control, lambda l: l.learners[0].npoints > 200)
    finally:
        shutil.rmtree(folder)


@run_with(Learner1D, Learner2D, LearnerND, AverageLearner,
    maybe_skip(SKOptLearner), IntegratorLearner)
def test_saving_with_datasaver(learner_type, f, learner_kwargs):
    f = generate_random_parametrization(f)
    g = lambda x: {'y': f(x), 't': random.random()}
    arg_picker = operator.itemgetter('y')
    learner = DataSaver(learner_type(g, **learner_kwargs), arg_picker)
    control = DataSaver(learner_type(g, **learner_kwargs), arg_picker)
    simple(learner, lambda l: l.npoints > 100)
    fd, path = tempfile.mkstemp()
    try:
        learner.save(path)
        control.load(path)
        if learner_type is not Learner1D:
            # Because different scales result in differnt losses
            np.testing.assert_almost_equal(learner.loss(), control.loss())
        assert learner.extra_data == control.extra_data

        # Try if the control is runnable
        simple(control, lambda l: l.npoints > 200)
    finally:
        os.remove(path)


@pytest.mark.xfail
@run_with(Learner1D, Learner2D, LearnerND)
def test_convergence_for_arbitrary_ordering(learner_type, f, learner_kwargs):
    """Learners that are learning the same function should converge
    to the same result "eventually" if given the same data, regardless
    of the order in which that data is given.
    """
    # XXX: not sure how to implement this. Can we say anything at all about
    #      the scaling of the loss with the number of points?
    raise NotImplementedError()


@pytest.mark.xfail
@run_with(Learner1D, Learner2D, LearnerND)
def test_learner_subdomain(learner_type, f, learner_kwargs):
    """Learners that never receive data outside of a subdomain should
       perform 'similarly' to learners defined on that subdomain only."""
    # XXX: not sure how to implement this. How do we measure "performance"?
    raise NotImplementedError()
