# -*- coding: utf-8 -*-

import asyncio

import pytest

from adaptive.learner import Learner1D, Learner2D
from adaptive.runner import (simple, BlockingRunner, AsyncRunner,
    SequentialExecutor, with_ipyparallel, with_distributed)


def blocking_runner(learner, goal):
    BlockingRunner(learner, goal, executor=SequentialExecutor())


def async_runner(learner, goal):
    runner = AsyncRunner(learner, goal, executor=SequentialExecutor())
    asyncio.get_event_loop().run_until_complete(runner.task)


runners = [simple, blocking_runner, async_runner]


def trivial_goal(learner):
    return learner.npoints > 10


@pytest.mark.parametrize('runner', runners)
def test_simple(runner):
    """Test that the runners actually run."""

    def f(x):
        return x

    learner = Learner1D(f, (-1, 1))
    runner(learner, lambda l: l.npoints > 10)
    assert len(learner.data) > 10


@pytest.mark.parametrize('runner', runners)
def test_nonconforming_output(runner):
    """Test that using a runner works with a 2D learner, even when the
    learned function outputs a 1-vector. This tests against the regression
    flagged in https://gitlab.kwant-project.org/qt/adaptive/issues/58.
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


### Test with different executors

@pytest.fixture(scope="session")
def ipyparallel_executor():
    from ipyparallel import Client
    import pexpect

    child = pexpect.spawn('ipcluster start -n 1')
    child.expect('Engines appear to have started successfully', timeout=35)
    yield Client()
    if not child.terminate(force=True):
        raise RuntimeError('Could not stop ipcluster')


@pytest.fixture(scope="session")
def dask_executor():
    from distributed import LocalCluster, Client

    client = Client(n_workers=1)
    yield client
    client.close()


def linear(x):
    return x


def test_concurrent_futures_executor():
    from concurrent.futures import ProcessPoolExecutor
    BlockingRunner(Learner1D(linear, (-1, 1)), trivial_goal,
                   executor=ProcessPoolExecutor(max_workers=1))


@pytest.mark.skipif(not with_ipyparallel, reason='IPyparallel is not installed')
def test_ipyparallel_executor(ipyparallel_executor):
    learner = Learner1D(linear, (-1, 1))
    BlockingRunner(learner, trivial_goal,
                   executor=ipyparallel_executor)
    assert learner.npoints > 0


@pytest.mark.skipif(not with_distributed, reason='dask.distributed is not installed')
def test_distributed_executor(dask_executor):
    learner = Learner1D(linear, (-1, 1))
    BlockingRunner(learner, trivial_goal,
                   executor=dask_executor)
    assert learner.npoints > 0
