from __future__ import annotations

import concurrent.futures as concurrent
import functools
import gzip
import inspect
import os
import pickle
import warnings
from collections.abc import Awaitable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import cloudpickle

if TYPE_CHECKING:
    from dask.distributed import Client as AsyncDaskClient


def named_product(**items: Sequence[Any]):
    names = items.keys()
    vals = items.values()
    return [dict(zip(names, res)) for res in product(*vals)]


@contextmanager
def restore(*learners) -> Iterator[None]:
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
    with _open(fname, "rb") as f:  # type: ignore[operator]
        return cloudpickle.load(f)


def copy_docstring_from(other: Callable) -> Callable:
    def decorator(method):
        method.__doc__ = other.__doc__
        return method

    return decorator


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
                " The DataFrame's value will be used.",
                stacklevel=2,
            )
    return functools.partial(function, **kwargs)


class SequentialExecutor(concurrent.Executor):
    """A trivial executor that runs functions synchronously.

    This executor is mainly for testing.
    """

    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.Future:  # type: ignore[override]
        fut: concurrent.Future = concurrent.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:
            fut.set_exception(e)
        return fut

    def map(self, fn, *iterable, timeout=None, chunksize=1):
        return map(fn, iterable)

    def shutdown(self, wait=True):
        pass


def _cache_key(args: tuple[Any], kwargs: dict[str, Any]) -> str:
    arg_strings = [str(a) for a in args]
    kwarg_strings = [f"{k}={v}" for k, v in sorted(kwargs.items())]
    return "_".join(arg_strings + kwarg_strings)


T = TypeVar("T")


def daskify(
    client: AsyncDaskClient, cache: bool = False
) -> Callable[[Callable[..., T]], Callable[..., Awaitable[T]]]:
    from dask import delayed

    def _daskify(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
        if cache:
            func.cache = {}  # type: ignore[attr-defined]

        delayed_func = delayed(func)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if cache:
                key = _cache_key(args, kwargs)  # type: ignore[arg-type]
                future = func.cache.get(key)  # type: ignore[attr-defined]

                if future is None:
                    future = client.compute(delayed_func(*args, **kwargs))
                    func.cache[key] = future  # type: ignore[attr-defined]
            else:
                future = client.compute(delayed_func(*args, **kwargs))

            result = await future
            return result

        return wrapper

    return _daskify
