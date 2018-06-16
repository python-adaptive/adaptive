# -*- coding: utf-8 -*-
from contextlib import contextmanager
from itertools import product
import time


class WithTime:
    def __init__(self, function, time=0):
        self.function = function
        self.time = time

    def __call__(self, *args, **kwargs):
        t_start = time.time()
        result = self.function(*args, **kwargs)
        self.time += time.time() - t_start
        return result

    def __getstate__(self):
        return (self.function, self.time)

    def __setstate__(self, state):
        self.__init__(*state)


class AverageTimeReturn:
    def __init__(self, function, total_time=0, n=0):
        self.function = function
        self.total_time = total_time
        self.n = n

    def __call__(self, *args, **kwargs):
        t_start = time.time()
        result = self.function(*args, **kwargs)
        self.total_time += time.time() - t_start
        self.n += 1
        return result, self.total_time / self.n

    def __getstate__(self):
        return (self.function, self.total_time, self.n)

    def __setstate__(self, state):
        self.__init__(*state)


class TimeReturn:
    def __init__(self, function, total_time=0):
        self.function = function
        self.total_time = total_time

    def __call__(self, *args, **kwargs):
        t_start = time.time()
        result = self.function(*args, **kwargs)
        self.total_time += time.time() - t_start
        return result, self.total_time

    def __getstate__(self):
        return (self.function, self.total_time)

    def __setstate__(self, state):
        self.__init__(*state)


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
