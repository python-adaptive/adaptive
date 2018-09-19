# -*- coding: utf-8 -*-

import random
import numpy as np

from ..learner import Learner1D
from ..runner import simple, replay_log


def test_learner1D_pending_loss_intervals():
    # https://gitlab.kwant-project.org/qt/adaptive/issues/99
    l = Learner1D(lambda x: x, (0, 4))

    l.tell(0, 0)
    l.tell(1, 0)
    l.tell(2, 0)
    assert set(l.losses_combined.keys()) == {(0, 1), (1, 2)}
    l.ask(1)
    assert set(l.losses_combined.keys()) == {(0, 1), (1, 2), (2, 4)}
    l.tell(3.5, 0)
    assert set(l.losses_combined.keys()) == {
        (0, 1), (1, 2), (2, 3.5), (3.5, 4.0)}


def test_learner1D_loss_interpolation_for_unasked_point():
    # https://gitlab.kwant-project.org/qt/adaptive/issues/99
    l = Learner1D(lambda x: x, (0, 4))

    l.tell(0, 0)
    l.tell(1, 0)
    l.tell(2, 0)

    assert l.ask(1) == ([4], [np.inf])
    assert l.losses == {(0, 1): 0.25, (1, 2): 0.25}
    assert l.losses_combined == {(0, 1): 0.25, (1, 2): 0.25, (2, 4.0): np.inf}

    # assert l.ask(1) == ([3], [np.inf])  # XXX: This doesn't return np.inf as loss_improvement...
    l.ask(1)
    assert l.losses == {(0, 1): 0.25, (1, 2): 0.25}
    assert l.losses_combined == {
        (0, 1): 0.25, (1, 2): 0.25, (2, 3.0): np.inf, (3.0, 4.0): np.inf}

    l.tell(4, 0)

    assert l.losses_combined == {
        (0, 1): 0.25, (1, 2): 0.25, (2, 3): 0.25, (3, 4): 0.25}


def test_learner1d_first_iteration():
    """Edge cases where we ask for a few points at the start."""
    learner = Learner1D(lambda x: None, (-1, 1))
    points, loss_improvements = learner.ask(2)
    assert set(points) == set(learner.bounds)

    learner = Learner1D(lambda x: None, (-1, 1))
    points, loss_improvements = learner.ask(3)
    assert set(points) == set([-1, 0, 1])

    learner = Learner1D(lambda x: None, (-1, 1))
    points, loss_improvements = learner.ask(1)
    assert len(points) == 1 and points[0] in learner.bounds
    rest = set([-1, 0, 1]) - set(points)
    points, loss_improvements = learner.ask(2)
    assert set(points) == set(rest)

    learner = Learner1D(lambda x: None, (-1, 1))
    points, loss_improvements = learner.ask(1)
    to_see = set(learner.bounds) - set(points)
    points, loss_improvements = learner.ask(1)
    assert set(points) == set(to_see)

    learner = Learner1D(lambda x: None, (-1, 1))
    learner.tell(1, 0)
    points, loss_improvements = learner.ask(1)
    assert points == [-1]

    learner = Learner1D(lambda x: None, (-1, 1))
    learner.tell(-1, 0)
    points, loss_improvements = learner.ask(1)
    assert points == [1]


def test_learner1d_loss_interpolation():
    learner = Learner1D(lambda _: 0, bounds=(-1, 1))

    learner.tell(-1, 0)
    learner.tell(1, 0)
    for i in range(100):
        # Add a 100 points with either None or 0
        if random.random() < 0.9:
            learner.tell(random.uniform(-1, 1), None)
        else:
            learner.tell(random.uniform(-1, 1), 0)

    for (x1, x2), loss in learner.losses_combined.items():
        expected_loss = (x2 - x1) / 2
        assert abs(expected_loss - loss) < 1e-15, (expected_loss, loss)


def _run_on_discontinuity(x_0, bounds):

    def f(x):
        return -1 if x < x_0 else +1

    learner = Learner1D(f, bounds)
    while learner.loss() > 0.1:
        (x,), _ = learner.ask(1)
        learner.tell(x, learner.function(x))

    return learner


def test_termination_on_discontinuities():

    learner = _run_on_discontinuity(0, (-1, 1))
    smallest_interval = min(abs(a - b) for a, b in learner.losses.keys())
    assert smallest_interval >= np.finfo(float).eps

    learner = _run_on_discontinuity(1, (-2, 2))
    smallest_interval = min(abs(a - b) for a, b in learner.losses.keys())
    assert smallest_interval >= np.finfo(float).eps

    learner = _run_on_discontinuity(0.5E3, (-1E3, 1E3))
    smallest_interval = min(abs(a - b) for a, b in learner.losses.keys())
    assert smallest_interval >= 0.5E3 * np.finfo(float).eps


def test_order_adding_points():
    # and https://gitlab.kwant-project.org/qt/adaptive/issues/98
    l = Learner1D(lambda x: x, (0, 1))
    l.tell_many([1, 0, 0.5], [0, 0, 0])
    assert l.losses_combined == {(0, 0.5): 0.5, (0.5, 1): 0.5}
    assert l.losses == {(0, 0.5): 0.5, (0.5, 1): 0.5}
    l.ask(1)


def test_adding_existing_point_passes_silently():
    # See https://gitlab.kwant-project.org/qt/adaptive/issues/97
    l = Learner1D(lambda x: x, (0, 4))
    l.tell(0, 0)
    l.tell(1, 0)
    l.tell(2, 0)
    l.tell(1, None)


def test_loss_at_machine_precision_interval_is_zero():
    """The loss of an interval smaller than _dx_eps
    should be set to zero."""
    def f(x):
        return 1 if x == 0 else 0

    def goal(l):
        return l.loss() < 0.01 or l.npoints >= 1000

    learner = Learner1D(f, bounds=(-1, 1))
    simple(learner, goal=goal)

    # this means loss < 0.01 was reached
    assert learner.npoints != 1000


def small_deviations(x):
    return 0 if x <= 1 else 1 + 10**(-random.randint(12, 14))


def test_small_deviations():
    """This tests whether the Learner1D can handle small deviations.
    See https://gitlab.kwant-project.org/qt/adaptive/merge_requests/73 and
    https://gitlab.kwant-project.org/qt/adaptive/issues/61."""

    eps = 5e-14
    learner = Learner1D(small_deviations, bounds=(1 - eps, 1 + eps))

    # Some non-determinism is needed to make this test fail so we keep
    # a list of points that will be evaluated later to emulate
    # parallel execution
    stash = []

    for i in range(100):
        xs, _ = learner.ask(10)

        # Save 5 random points out of `xs` for later
        random.shuffle(xs)
        for _ in range(5):
            stash.append(xs.pop())

        for x in xs:
            learner.tell(x, learner.function(x))

        # Evaluate and add 5 random points from `stash`
        random.shuffle(stash)
        for _ in range(5):
            learner.tell(stash.pop(), learner.function(x))

        if learner.loss() == 0:
            # If this condition is met, the learner can't return any
            # more points.
            break
