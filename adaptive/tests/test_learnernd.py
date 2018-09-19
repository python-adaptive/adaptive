# -*- coding: utf-8 -*-

from ..learner import LearnerND
from ..runner import replay_log


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
