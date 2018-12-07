# -*- coding: utf-8 -*-

import scipy.spatial

from adaptive.learner import LearnerND
from adaptive.runner import replay_log, simple

from .test_learners import ring_of_fire, generate_random_parametrization


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
