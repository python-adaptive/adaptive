# -*- coding: utf-8 -*-

import numpy as np
import pytest
from ..learner import IntegratorLearner
from ..learner.integrator_learner import DivergentIntegralError
from .algorithm_4 import algorithm_4, f0, f7, f21, f24, f63, fdiv
from .algorithm_4 import DivergentIntegralError as A4DivergentIntegralError


def run_integrator_learner(f, a, b, tol, nr_points):
    learner = IntegratorLearner(f, bounds=(a, b), tol=tol)
    for _ in range(nr_points):
        points, _ = learner.choose_points(1)
        learner.add_data(points, map(learner.function, points))
    return learner


def same_ivals(f, a, b, tol):
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)

        learner = run_integrator_learner(f, a, b, tol, nr_points)

        # This will only show up if the test fails, anyway
        print('igral difference', learner.igral-igral,
              'err difference', learner.err - err)

        return learner.equal(ivals, verbose=True)


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

    assert learner.equal(ivals, verbose=True)


def test_machine_precision2():
    f, a, b, tol = [f63, 0, 1, 1e-10]
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
