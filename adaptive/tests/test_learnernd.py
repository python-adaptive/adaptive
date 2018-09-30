# -*- coding: utf-8 -*-

from ..learner import LearnerND
from ..runner import replay_log, BlockingRunner
import numpy as np


def sphere(xyz):
    x, y, z = xyz
    a = 0.4
    return x + z**2 + np.exp(-(x**2 + y**2 + z**2 - 0.75**2)**2/a**4)


def test_failure_case_LearnerND():
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


def test_anisotropic_3d():
    # there was this bug that the total volume would exceed the bounding box 
    # volume for the anisotropic 3d learnerND
    # learner = adaptive.LearnerND(ring, bounds=[(-1, 1), (-1, 1)], anisotropic=True)
    learner = LearnerND(sphere, bounds=[(-1, 1), (-1, 1), (-1, 1)], anisotropic=True)
    def goal(l):
        if l.tri:
            sum_of_simplex_volumes = np.sum(l.tri.volumes())
            assert sum_of_simplex_volumes < 8.00000000001
        return l.npoints >= 1000
    BlockingRunner(learner, goal, ntasks=1)

    assert learner.npoints >= 1000
