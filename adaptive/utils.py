# -*- coding: utf-8 -*-
from contextlib import contextmanager
from functools import wraps
from itertools import product
import time


def timed(f, *args, **kwargs):
    t_start = time.time()
    result = f(*args, **kwargs)
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


def cache_latest(f):
    """Cache the latest return value of the function and add it
    as 'self._cache[f.__name__]'."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[f.__name__] = f(*args, **kwargs)
        return self._cache[f.__name__]
    return wrapper
