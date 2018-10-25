# -*- coding: utf-8 -*-

import asyncio

import pytest

from ..learner import Learner1D, Learner2D
from ..runner import simple, BlockingRunner, AsyncRunner, SequentialExecutor


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
