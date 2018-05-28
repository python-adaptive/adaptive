# -*- coding: utf-8 -*-

import concurrent.futures as concurrent
import asyncio

from ..learner import Learner2D
from ..runner import simple, BlockingRunner, AsyncRunner, SequentialExecutor



def test_nonconforming_output():
    """Test that using a runner works with a 2D learner, even when the
    learned function outputs a 1-vector. This tests against the regression
    flagged in https://gitlab.kwant-project.org/qt/adaptive/issues/58.
    """

    def f(x):
        return [0]

    def goal(l):
        return l.npoints > 1

    learner = Learner2D(f, [(-1, 1), (-1, 1)])

    simple(learner, goal)
    BlockingRunner(learner, goal, executor=SequentialExecutor())
    runner = AsyncRunner(learner, goal, executor=SequentialExecutor())
    asyncio.get_event_loop().run_until_complete(runner.task)
