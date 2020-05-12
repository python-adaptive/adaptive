import asyncio
import os
import sys
import time

import flaky
import pytest

from adaptive.learner import Learner1D, Learner2D
from adaptive.runner import (
    AsyncRunner,
    BlockingRunner,
    SequentialExecutor,
    simple,
    stop_after,
    with_distributed,
    with_ipyparallel,
    with_loky,
)


def blocking_runner(learner, goal):
    BlockingRunner(learner, goal, executor=SequentialExecutor())


def async_runner(learner, goal):
    runner = AsyncRunner(learner, goal, executor=SequentialExecutor())
    asyncio.get_event_loop().run_until_complete(runner.task)


runners = [simple, blocking_runner, async_runner]


def trivial_goal(learner):
    return learner.npoints > 10


@pytest.mark.parametrize("runner", runners)
def test_simple(runner):
    """Test that the runners actually run."""

    def f(x):
        return x

    learner = Learner1D(f, (-1, 1))
    runner(learner, lambda l: l.npoints > 10)
    assert len(learner.data) > 10


@pytest.mark.parametrize("runner", runners)
def test_nonconforming_output(runner):
    """Test that using a runner works with a 2D learner, even when the
    learned function outputs a 1-vector. This tests against the regression
    flagged in https://github.com/python-adaptive/adaptive/issues/81.
    """

    def f(x):
        return [0]

    runner(Learner2D(f, [(-1, 1), (-1, 1)]), trivial_goal)


def test_aync_def_function():
    async def f(x):
        return x

    learner = Learner1D(f, (-1, 1))
    runner = AsyncRunner(learner, trivial_goal)
    asyncio.get_event_loop().run_until_complete(runner.task)


# --- Test with different executors


@pytest.fixture(scope="session")
def ipyparallel_executor():
    from ipyparallel import Client

    if os.name == "nt":
        import wexpect as expect
    else:
        import pexpect as expect

    child = expect.spawn("ipcluster start -n 1")
    child.expect("Engines appear to have started successfully", timeout=35)
    yield Client()
    if not child.terminate(force=True):
        raise RuntimeError("Could not stop ipcluster")


@pytest.fixture(scope="session")
def loky_executor():
    import loky

    return loky.get_reusable_executor()


def linear(x):
    return x


def test_concurrent_futures_executor():
    from concurrent.futures import ProcessPoolExecutor

    BlockingRunner(
        Learner1D(linear, (-1, 1)),
        trivial_goal,
        executor=ProcessPoolExecutor(max_workers=1),
    )


def test_stop_after_goal():
    seconds_to_wait = 0.2  # don't make this too large or the test will take ages
    start_time = time.time()
    BlockingRunner(Learner1D(linear, (-1, 1)), stop_after(seconds=seconds_to_wait))
    stop_time = time.time()
    assert stop_time - start_time > seconds_to_wait


@pytest.mark.skipif(not with_ipyparallel, reason="IPyparallel is not installed")
def test_ipyparallel_executor(ipyparallel_executor):
    learner = Learner1D(linear, (-1, 1))
    BlockingRunner(learner, trivial_goal, executor=ipyparallel_executor)
    assert learner.npoints > 0


@flaky.flaky(max_runs=5)
@pytest.mark.timeout(60)
@pytest.mark.skipif(not with_distributed, reason="dask.distributed is not installed")
@pytest.mark.skipif(os.name == "nt", reason="XXX: seems to always fail")
@pytest.mark.skipif(sys.platform == "darwin", reason="XXX: intermittently fails")
def test_distributed_executor():
    from distributed import Client

    learner = Learner1D(linear, (-1, 1))
    client = Client(n_workers=1)
    BlockingRunner(learner, trivial_goal, executor=client)
    client.shutdown()
    assert learner.npoints > 0


@pytest.mark.skipif(not with_loky, reason="loky not installed")
def test_loky_executor(loky_executor):
    learner = Learner1D(lambda x: x, (-1, 1))
    BlockingRunner(
        learner, trivial_goal, executor=loky_executor, shutdown_executor=True
    )
    assert learner.npoints > 0


def test_default_executor():
    learner = Learner1D(linear, (-1, 1))
    runner = AsyncRunner(learner, goal=lambda l: l.npoints > 10)
    asyncio.get_event_loop().run_until_complete(runner.task)
