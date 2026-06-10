"""Make ``flaky`` reruns work with ``pytest-randomly``."""

import functools as ft
import random
from collections import Counter

import numpy as np

_attempts: Counter = Counter()


def fresh_seed_each_run(func):
    """Make ``@flaky.flaky`` reruns draw new random values.

    ``pytest-randomly`` reseeds the global RNG at the start of every test
    call phase — including reruns triggered by the ``flaky`` plugin — so a
    randomized test that fails for the session seed fails identically on
    every rerun, making the retries useless. Reseeding cannot happen in a
    fixture (``pytest-randomly`` reseeds in ``pytest_runtest_call``, after
    fixture setup), so this mixes the attempt number into the seed at the
    start of the test call itself. Each rerun gets new draws while the
    whole sequence stays reproducible via ``--randomly-seed``.

    Apply directly on the test function, below ``@flaky.flaky`` and any
    parametrization.
    """

    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        key = (func.__qualname__, repr(args), repr(kwargs))
        attempt = _attempts[key]
        _attempts[key] += 1
        if attempt:
            seed = (random.getrandbits(32) + attempt) % 2**32
            random.seed(seed)
            np.random.seed(seed)
        return func(*args, **kwargs)

    return wrapper
