from __future__ import annotations

import abc
import functools
import gzip
import inspect
import os
import pickle
import warnings
from contextlib import _GeneratorContextManager, contextmanager
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import cloudpickle


def named_product(**items: Mapping[str, Sequence[Any]]):
    names = items.keys()
    vals = items.values()
    return [dict(zip(names, res)) for res in product(*vals)]


@contextmanager
def restore(*learners) -> _GeneratorContextManager:
    states = [learner.__getstate__() for learner in learners]
    try:
        yield
    finally:
        for state, learner in zip(states, learners):
            learner.__setstate__(state)


def cache_latest(f: Callable) -> Callable:
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


def save(fname: str, data: Any, compress: bool = True) -> bool:
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


def load(fname: str, compress: bool = True) -> Any:
    fname = os.path.expanduser(fname)
    _open = gzip.open if compress else open
    with _open(fname, "rb") as f:
        return cloudpickle.load(f)


def copy_docstring_from(other: Callable) -> Callable:
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


def _default_parameters(function, function_prefix: str = "function."):
    sig = inspect.signature(function)
    defaults = {
        f"{function_prefix}{k}": v.default
        for i, (k, v) in enumerate(sig.parameters.items())
        if v.default != inspect._empty and i >= 1
    }
    return defaults


def assign_defaults(function, df, function_prefix: str = "function."):
    defaults = _default_parameters(function, function_prefix)
    for k, v in defaults.items():
        df[k] = len(df) * [v]
        df[k] = df[k].astype("category")


def partial_function_from_dataframe(function, df, function_prefix: str = "function."):
    if function_prefix == "":
        raise ValueError(
            "The function_prefix cannot be an empty string because"
            " it is used to distinguish between function and learner parameters."
        )
    kwargs = {}
    for col in df.columns:
        if col.startswith(function_prefix):
            k = col.split(function_prefix, 1)[1]
            vs = df[col]
            v, *rest = vs.unique()
            if rest:
                raise ValueError(f"The column '{col}' can only have one value.")
            kwargs[k] = v
    if not kwargs:
        return function

    sig = inspect.signature(function)
    for k, v in kwargs.items():
        if k not in sig.parameters:
            raise ValueError(
                f"The DataFrame contains a default parameter"
                f" ({k}={v}) but the function does not have that parameter."
            )
        default = sig.parameters[k].default
        if default != inspect._empty and kwargs[k] != default:
            warnings.warn(
                f"The DataFrame contains a default parameter"
                f" ({k}={v}) but the function already has a default ({k}={default})."
                " The DataFrame's value will be used."
            )
    return functools.partial(function, **kwargs)
