# -*- coding: utf-8 -*-

import random

import numpy as np

from adaptive.learner import AverageLearner


def test_only_returns_new_points():
    learner = AverageLearner(lambda x: x, atol=None, rtol=0.01)

    # Only tell it n = 5...10
    for i in range(5, 10):
        learner.tell(i, 1)

    learner.tell_pending(0)  # This means it shouldn't return 0 anymore

    assert learner.ask(1)[0][0] == 1
    assert learner.ask(1)[0][0] == 2
    assert learner.ask(1)[0][0] == 3
    assert learner.ask(1)[0][0] == 4
    assert learner.ask(1)[0][0] == 10


def test_avg_std_and_npoints():
    learner = AverageLearner(lambda x: x, atol=None, rtol=0.01)

    for i in range(300):
        # This will add 5000 points at random values of n.
        # It could try to readd already evaluated points.

        n = random.randint(0, 2 * 300)
        value = random.random()

        # With 10% chance None is added to simulate asking that point.
        if value < 0.9:
            learner.tell(n, value)
        else:
            learner.tell_pending(n)

        if i > 2 and i % 10 == 0:
            # We need more than two points for 'learner.std' to be defined.
            values = np.array(list(learner.data.values()))
            std = np.sqrt(sum((values - values.mean())**2) / (len(values) - 1))
            assert learner.npoints == len(learner.data)
            assert abs(learner.sum_f - values.sum()) < 1e-13
            assert abs(learner.std - std) < 1e-13
