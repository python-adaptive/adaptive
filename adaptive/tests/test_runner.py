import platform
import sys
import time

import numpy as np
import pytest

from adaptive.learner import (
    BalancingLearner,
    DataSaver,
    IntegratorLearner,
    Learner1D,
    Learner2D,
    SequenceLearner,
)
from adaptive.runner import (
    AsyncRunner,
    BlockingRunner,
    SequentialExecutor,
    auto_goal,
    simple,
    stop_after,
    with_distributed,
    with_ipyparallel,
)

OPERATING_SYSTEM = platform.system()


def blocking_runner(learner, **kw):
    BlockingRunner(learner, executor=SequentialExecutor(), **kw)


def async_runner(learner, **kw):
    runner = AsyncRunner(learner, executor=SequentialExecutor(), **kw)
    runner.block_until_done()


runners = [simple, blocking_runner, async_runner]


@pytest.mark.parametrize("runner", runners)
def test_simple(runner):
    """Test that the runners actually run."""

    def f(x):
        return x

    learner = Learner1D(f, (-1, 1))
    runner(learner, npoints_goal=10)
    assert len(learner.data) >= 10


@pytest.mark.parametrize("runner", runners)
def test_nonconforming_output(runner):
    """Test that using a runner works with a 2D learner, even when the
    learned function outputs a 1-vector. This tests against the regression
    flagged in https://github.com/python-adaptive/adaptive/issues/81.
    """

    def f(x):
        return [0]

    runner(Learner2D(f, ((-1, 1), (-1, 1))), npoints_goal=10)


def test_aync_def_function():
    async def f(x):
        return x

    learner = Learner1D(f, (-1, 1))
    runner = AsyncRunner(learner, npoints_goal=10)
    runner.block_until_done()


# --- Test with different executors


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
        npoints_goal=10,
        executor=ProcessPoolExecutor(max_workers=1),
    )


def test_stop_after_goal():
    seconds_to_wait = 0.2  # don't make this too large or the test will take ages
    start_time = time.time()
    BlockingRunner(Learner1D(linear, (-1, 1)), goal=stop_after(seconds=seconds_to_wait))
    stop_time = time.time()
    assert stop_time - start_time > seconds_to_wait


@pytest.mark.skipif(not with_ipyparallel, reason="IPyparallel is not installed")
@pytest.mark.skipif(
    OPERATING_SYSTEM == "Windows" and sys.version_info >= (3, 7),
    reason="Gets stuck in CI",
)
@pytest.mark.skipif(OPERATING_SYSTEM == "Darwin", reason="Cannot stop ipcluster")
def test_ipyparallel_executor():
    from ipyparallel import Client

    if OPERATING_SYSTEM == "Windows":
        import wexpect as expect
    else:
        import pexpect as expect

    child = expect.spawn("ipcluster start -n 1")
    child.expect("Engines appear to have started successfully", timeout=35)
    ipyparallel_executor = Client()
    learner = Learner1D(linear, (-1, 1))
    BlockingRunner(learner, npoints_goal=10, executor=ipyparallel_executor)

    assert learner.npoints > 0

    if not child.terminate(force=True):
        raise RuntimeError("Could not stop ipcluster")


@pytest.mark.timeout(60)
@pytest.mark.skipif(not with_distributed, reason="dask.distributed is not installed")
@pytest.mark.skipif(OPERATING_SYSTEM == "Windows", reason="XXX: seems to always fail")
@pytest.mark.skipif(OPERATING_SYSTEM == "Darwin", reason="XXX: intermittently fails")
@pytest.mark.skipif(OPERATING_SYSTEM == "Linux", reason="XXX: intermittently fails")
def test_distributed_executor():
    from distributed import Client

    learner = Learner1D(linear, (-1, 1))
    client = Client(n_workers=1)
    BlockingRunner(learner, npoints_goal=10, executor=client)
    client.shutdown()
    assert learner.npoints > 0


def test_loky_executor(loky_executor):
    learner = Learner1D(lambda x: x, (-1, 1))
    BlockingRunner(
        learner, npoints_goal=10, executor=loky_executor, shutdown_executor=True
    )
    assert learner.npoints > 0


def test_default_executor():
    learner = Learner1D(linear, (-1, 1))
    runner = AsyncRunner(learner, npoints_goal=10)
    runner.block_until_done()


def test_auto_goal():
    learner = Learner1D(linear, (-1, 1))
    simple(learner, auto_goal(npoints=4))
    assert learner.npoints == 4

    learner = Learner1D(linear, (-1, 1))
    simple(learner, auto_goal(loss=0.5))
    assert learner.loss() <= 0.5

    learner = SequenceLearner(linear, np.linspace(-1, 1))
    simple(learner, auto_goal(learner=learner))
    assert learner.done()

    learner = IntegratorLearner(linear, bounds=(0, 1), tol=0.1)
    simple(learner, auto_goal(learner=learner))
    assert learner.done()

    learner = Learner1D(linear, (-1, 1))
    learner = DataSaver(learner, lambda x: x)
    simple(learner, auto_goal(npoints=4, learner=learner))
    assert learner.npoints == 4

    learner1 = Learner1D(linear, (-1, 1))
    learner2 = Learner1D(linear, (-2, 2))
    balancing_learner = BalancingLearner([learner1, learner2])
    simple(balancing_learner, auto_goal(npoints=4, learner=balancing_learner))
    assert learner1.npoints == 4 and learner2.npoints == 4

    learner1 = Learner1D(linear, bounds=(0, 1))
    learner1 = DataSaver(learner1, lambda x: x)
    learner2 = Learner1D(linear, bounds=(0, 1))
    learner2 = DataSaver(learner2, lambda x: x)
    balancing_learner = BalancingLearner([learner1, learner2])
    simple(balancing_learner, auto_goal(npoints=10, learner=balancing_learner))
    assert learner1.npoints == 10 and learner2.npoints == 10

    learner = Learner1D(linear, (-1, 1))
    t_start = time.time()
    simple(learner, auto_goal(duration=1e-2, learner=learner))
    t_end = time.time()
    assert t_end - t_start >= 1e-2
