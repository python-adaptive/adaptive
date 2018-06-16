# -*- coding: utf-8 -*-
from contextlib import contextmanager
from itertools import product
import time


class TimeReturn:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        t_start = time.time()
        result = self.function(*args, **kwargs)
        return result, time.time() - t_start


def named_product(**items):
    names = items.keys()
    vals = items.values()
    return [dict(zip(names, res)) for res in product(*vals)]


@contextmanager
def restore(*learners):
    states = [learner.__getstate__() for learner in learners]
    try:
        yield
    finally:
        for state, learner in zip(states, learners):
            learner.__setstate__(state)
