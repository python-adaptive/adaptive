# -*- coding: utf-8 -*-
from functools import partial
from operator import attrgetter

import numpy as np
import pytest
from ..learner import IntegratorLearner
from ..learner.integrator_learner import DivergentIntegralError
from .algorithm_4 import algorithm_4, f0, f7, f21, f24, f63, fdiv
from .algorithm_4 import DivergentIntegralError as A4DivergentIntegralError

eps = np.spacing(1)

def run_integrator_learner(f, a, b, tol, nr_points):
    learner = IntegratorLearner(f, bounds=(a, b), tol=tol)
    for _ in range(nr_points):
        points, _ = learner.choose_points(1)
        learner.add_data(points, map(learner.function, points))
    return learner


def equal_ival(ival, other, *, verbose=False):
    """Note: Implementing __eq__ breaks SortedContainers in some way."""
    if not ival.complete:
        if verbose:
            print('Interval {} is not complete.'.format(ival))
        return False

    slots = set(ival.__slots__).intersection(other.__slots__)
    same_slots = []
    for s in slots:
        a = getattr(ival, s)
        b = getattr(other, s)
        is_equal = np.allclose(a, b, rtol=0, atol=eps, equal_nan=True)
        if verbose and not is_equal:
            print('ival.{} - other.{} = {}'.format(s, s, a - b))
        same_slots.append(is_equal)

    return all(same_slots)

def equal_ivals(ivals, other, *, verbose=False):
    """Note: `other` is a list of ivals."""
    if len(ivals) != len(other):
        if verbose:
            print('len(ivals)={} != len(other)={}'.format(
                len(ivals), len(other)))
        return False

    ivals = [sorted(i, key=attrgetter('a')) for i in [ivals, other]]
    return all(equal_ival(ival, other_ival, verbose=verbose)
               for ival, other_ival in zip(*ivals))

def same_ivals(f, a, b, tol):
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)

        learner = run_integrator_learner(f, a, b, tol, nr_points)

        # This will only show up if the test fails, anyway
        print('igral difference', learner.igral-igral,
              'err difference', learner.err - err)

        return equal_ivals(learner.ivals, ivals, verbose=True)


def test_cquad():
    for i, args in enumerate([[f0, 0, 3, 1e-5],
                              [f7, 0, 1, 1e-6],
                              [f21, 0, 1, 1e-3],
                              [f24, 0, 3, 1e-3]]):
        assert same_ivals(*args), 'Function {}'.format(i)


@pytest.mark.xfail
def test_machine_precision():
    f, a, b, tol = [f63, 0, 1, 1e-10]
    igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)

    learner = run_integrator_learner(f, a, b, tol, nr_points)

    print('igral difference', learner.igral-igral,
          'err difference', learner.err - err)

    assert equal_ivals(learner.ivals, ivals, verbose=True)


def test_machine_precision2():
    f, a, b, tol = [partial(f63, u=0.987654321, e=0.45), 0, 1, 1e-10]
    igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)

    learner = run_integrator_learner(f, a, b, tol, nr_points)

    np.testing.assert_almost_equal(igral, learner.igral)
    np.testing.assert_almost_equal(err, learner.err)


def test_divergence():
    """This function should raise a DivergentIntegralError."""
    f, a, b, tol = fdiv, 0, 1, 1e-6
    with pytest.raises(A4DivergentIntegralError) as e:
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)

    nr_points = e.value.nr_points

    with pytest.raises(DivergentIntegralError):
        learner = run_integrator_learner(f, a, b, tol, nr_points)
