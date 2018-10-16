# -*- coding: utf-8 -*-

import random

import numpy as np
import pytest
import scipy.spatial

from adaptive.learner import LearnerND
from adaptive.runner import replay_log, simple
from adaptive.tests.test_learners import (
    ring_of_fire, generate_random_parametrization)


def test_faiure_case_LearnerND():
    log = [
        ('ask', 4),
        ('tell', (-1, -1, -1), 1.607873907219222e-101),
        ('tell', (-1, -1, 1), 1.607873907219222e-101),
        ('ask', 2),
        ('tell', (-1, 1, -1), 1.607873907219222e-101),
        ('tell', (-1, 1, 1), 1.607873907219222e-101),
        ('ask', 2),
        ('tell', (1, -1, 1), 2.0),
        ('tell', (1, -1, -1), 2.0),
        ('ask', 2),
        ('tell', (0.0, 0.0, 0.0), 4.288304431237686e-06),
        ('tell', (1, 1, -1), 2.0)
    ]
    learner = LearnerND(lambda *x: x, bounds=[(-1, 1), (-1, 1), (-1, 1)])
    replay_log(learner, log)


def test_interior_vs_bbox_gives_same_result():
    f = generate_random_parametrization(ring_of_fire)
    
    control = LearnerND(f, bounds=[(-1, 1), (-1, 1)])
    hull = scipy.spatial.ConvexHull(control._bounds_points)
    learner = LearnerND(f, bounds=hull)

    simple(control, goal=lambda l: l.loss() < 0.1)
    simple(learner, goal=lambda l: l.loss() < 0.1)

    assert learner.data == control.data


# This sometimes fails and sometimes succeeds, my guess would be that this could
# be due to a numerical precision error:
# In the very beginning the loss of every interval is the same (as the function
# is highly symetric), then by machine precision there will be some error and
# then the simplex that has by accident some error that reduces the loss,
# will be chosen.
@pytest.mark.xfail
def test_learner_performance_is_invariant_under_scaling():
    kwargs = dict(bounds=[(-1, 1)]*2)
    f = generate_random_parametrization(ring_of_fire)

    control = LearnerND(f, **kwargs)

    xscale = 1000 * random.random()
    yscale = 1000 * random.random()

    l_kwargs = dict(kwargs)
    l_kwargs['bounds'] = xscale * np.array(l_kwargs['bounds'])
    learner = LearnerND(lambda x: yscale * f(np.array(x) / xscale), **l_kwargs)

    control._recompute_losses_factor = 1
    learner._recompute_losses_factor = 1

    npoints = random.randrange(1000, 2000)

    for n in range(npoints):
        cxs, _ = control.ask(1)
        xs, _ = learner.ask(1)

        control.tell_many(cxs, [control.function(x) for x in cxs])
        learner.tell_many(xs , [learner.function(x) for x in xs])
        if n > 100:
            assert np.isclose(learner.loss(), control.loss())


def test_learner_loss_is_invariant_under_scaling():
    kwargs = dict(bounds=[(-1, 1)]*2)
    f = generate_random_parametrization(ring_of_fire)

    control = LearnerND(f, **kwargs)

    xscale = 1000 * random.random()
    yscale = 1000 * random.random()

    l_kwargs = dict(kwargs)
    l_kwargs['bounds'] = xscale * np.array(l_kwargs['bounds'])
    learner = LearnerND(lambda x: yscale * f(np.array(x) / xscale), **l_kwargs)

    control._recompute_losses_factor = 1
    learner._recompute_losses_factor = 1

    npoints = 1000

    for n in range(npoints):
        cx = control.ask(1)[0][0]
        x = np.array(cx) * xscale
        control.tell(cx, control.function(cx))
        learner.tell(x , learner.function(x))
        if n > 5:
            assert np.isclose(learner.loss(), control.loss())
            assert learner.tri.simplices == control.tri.simplices
            if n % 29 == 0: # since this is very slow, only do it every now and then
                for simplex in learner.tri.simplices:
                    assert np.isclose(learner.losses()[simplex], control.losses()[simplex])
