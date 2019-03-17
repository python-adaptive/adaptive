# -*- coding: utf-8 -*-

from adaptive.learner import Learner1D, BalancingLearner
from adaptive.runner import simple


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


def test_distribute_first_points_over_learners():
    for strategy in ['loss', 'loss_improvements', 'npoints']:
        learners = [Learner1D(lambda x: x, bounds=(-1, 1)) for i in range(10)]
        learner = BalancingLearner(learners, strategy=strategy)
        points, _ = learner.ask(100)
        i_learner, xs = zip(*points)
        # assert that are all learners in the suggested points
        assert len(set(i_learner)) == len(learners)


def test_strategies():
    goals = {
        'loss': lambda l: l.loss() < 0.1,
        'loss_improvements': lambda l: l.loss() < 0.1,
        'npoints': lambda bl: all(l.npoints > 10 for l in bl.learners)
    }

    for strategy, goal in goals.items():
        learners = [Learner1D(lambda x: x, bounds=(-1, 1)) for i in range(10)]
        learner = BalancingLearner(learners, strategy=strategy)
        simple(learner, goal=goal)
