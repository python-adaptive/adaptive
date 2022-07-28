import abc
import functools
import gzip
import os
import pickle
from contextlib import contextmanager
from itertools import product

import cloudpickle


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
        if not hasattr(self, "_cache"):
            self._cache = {}
        self._cache[f.__name__] = f(*args, **kwargs)
        return self._cache[f.__name__]

    return wrapper


def save(fname, data, compress=True):
    fname = os.path.expanduser(fname)
    dirname = os.path.dirname(fname)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    blob = cloudpickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    if compress:
        blob = gzip.compress(blob)

    temp_file = f"{fname}.{os.getpid()}"

    try:
        with open(temp_file, "wb") as f:
            f.write(blob)
    except OSError:
        return False

    try:
        os.replace(temp_file, fname)
    except OSError:
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return True


def load(fname, compress=True):
    fname = os.path.expanduser(fname)
    _open = gzip.open if compress else open
    with _open(fname, "rb") as f:
        return cloudpickle.load(f)


def copy_docstring_from(other):
    def decorator(method):
        return functools.wraps(other)(method)

    return decorator


class _RequireAttrsABCMeta(abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        for name, type_ in obj.__annotations__.items():
            try:
                x = getattr(obj, name)
            except AttributeError:
                raise AttributeError(
                    f"Required attribute {name} not set in __init__."
                ) from None
            else:
                if not isinstance(x, type_):
                    msg = f"The attribute '{name}' should be of type {type_}, not {type(x)}."
                    raise TypeError(msg)
        return obj
