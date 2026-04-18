from __future__ import annotations

import functools as ft
import math
import random

import numpy as np
import pytest

from adaptive.learner import BalancingLearner, Learner1D, Learner2D
from adaptive.runner import simple

strategies = ["loss", "loss_improvements", "npoints", "cycle"]


def ring_of_fire(xy, d):
    a = 0.2
    x, y = xy
    return x + math.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


def test_balancing_learner_loss_cache():
    learner = Learner1D(lambda x: x, bounds=(-1, 1))
    learner.tell(-1, -1)
    learner.tell(1, 1)
    learner.tell_pending(0)

    real_loss = learner.loss(real=True)
    pending_loss = learner.loss(real=False)

    # Test if the real and pending loss are cached correctly
    bl = BalancingLearner([learner])
    assert bl.loss(real=True) == real_loss
    assert bl.loss(real=False) == pending_loss

    # Test if everything is still fine when executed in the reverse order
    bl = BalancingLearner([learner])
    assert bl.loss(real=False) == pending_loss
    assert bl.loss(real=True) == real_loss


@pytest.mark.parametrize("strategy", strategies)
def test_distribute_first_points_over_learners(strategy):
    for initial_points in [0, 3]:
        learners = [Learner1D(lambda x: x, bounds=(-1, 1)) for i in range(10)]
        learner = BalancingLearner(learners, strategy=strategy)

        points = learner.ask(initial_points)[0]
        learner.tell_many(points, [x for i, x in points])

        points, _ = learner.ask(100)
        i_learner, xs = zip(*points)
        # assert that are all learners in the suggested points
        assert len(set(i_learner)) == len(learners)


@pytest.mark.parametrize("strategy", strategies)
def test_ask_0(strategy):
    learners = [Learner1D(lambda x: x, bounds=(-1, 1)) for i in range(10)]
    learner = BalancingLearner(learners, strategy=strategy)
    points, _ = learner.ask(0)
    assert len(points) == 0


@pytest.mark.parametrize(
    "strategy, goal_type, goal",
    [
        ("loss", "loss_goal", 0.1),
        ("loss_improvements", "loss_goal", 0.1),
        ("npoints", "goal", lambda bl: all(lrn.npoints > 10 for lrn in bl.learners)),
        ("cycle", "loss_goal", 0.1),
    ],
)
def test_strategies(strategy, goal_type, goal):
    learners = [Learner1D(lambda x: x, bounds=(-1, 1)) for i in range(10)]
    learner = BalancingLearner(learners, strategy=strategy)
    simple(learner, **{goal_type: goal})


def test_loss_improvements_strategy_with_tell_pending_false_reserves_child_points():
    random.seed(3104322362)
    np.random.seed(3104322362 % 2**32)

    learners = [
        Learner2D(
            ft.partial(ring_of_fire, d=random.uniform(0.2, 1)),
            bounds=((-1, 1), (-1, 1)),
        )
        for _ in range(4)
    ]
    learner = BalancingLearner(learners, strategy="loss_improvements")

    stash = []
    for n, m in [(1, 1), (4, 4), (2, 0), (4, 4), (8, 6)]:
        xs, _ = learner.ask(n, tell_pending=False)
        random.shuffle(xs)
        for _ in range(m):
            stash.append(xs.pop())

        for x in xs:
            learner.tell(x, learner.function(x))

        random.shuffle(stash)
        for _ in range(m):
            x = stash.pop()
            learner.tell(x, learner.function(x))

    assert all(not child.pending_points for child in learners)
