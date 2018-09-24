# -*- coding: utf-8 -*-

from ..learner import AverageLearner


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
