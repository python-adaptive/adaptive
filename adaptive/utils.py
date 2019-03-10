# -*- coding: utf-8 -*-

from contextlib import contextmanager
import functools
import gzip
from itertools import product
import os
import pickle
import time


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
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[f.__name__] = f(*args, **kwargs)
        return self._cache[f.__name__]
    return wrapper


def save(fname, data, compress=True):
    fname = os.path.expanduser(fname)
    dirname = os.path.dirname(fname)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    _open = gzip.open if compress else open
    with _open(fname, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(fname, compress=True):
    fname = os.path.expanduser(fname)
    _open = gzip.open if compress else open
    with _open(fname, 'rb') as f:
        return pickle.load(f)


def copy_docstring_from(other):
    def decorator(method):
        return functools.wraps(other)(method)
    return decorator
