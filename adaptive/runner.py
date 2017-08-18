# -*- coding: utf-8 -*-
import asyncio
import concurrent.futures as concurrent

import ipyparallel


class Runner:
    """Runs a learning algorithm in an executor.

    Parameters
    ----------
    learner : Learner
    executor : concurrent.futures.Executor, or ipyparallel.Client, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new ProcessPoolExecutor is used.
    goal : callable, optional
        The end condition for the calculation. This function must take the
        learner as its sole argument, and return True if we should stop.
    ioloop : asyncio.AbstractEventLoop, optional
        The ioloop in which to run the learning algorithm. If not provided,
        the default event loop is used.

    Attributes
    ----------
    task : asyncio.Task
        The underlying task. May be cancelled to stop the runner.
    learner : Learner
        The underlying learner. May be queried for its state
    """

    def __init__(self, learner, executor=None, goal=None, *, ioloop=None):
        ioloop = ioloop if ioloop else asyncio.get_event_loop()
        self.executor = _ensure_async_executor(executor, ioloop)
        self.learner = learner

        if goal is None:
            def goal(_):
                return True

        coro = self._run(self.learner, self.executor, goal, ioloop)
        self.task = ioloop.create_task(coro)

    @staticmethod
    async def _run(learner, executor, goal, ioloop):
        first_completed = asyncio.FIRST_COMPLETED
        xs = dict()
        done = [None] * _get_executor_ncores(executor)

        try:
            while not goal(learner):
                # Launch tasks to replace the ones that completed
                # on the last iteration.
                for x in learner.choose_points(len(done)):
                    xs[executor.submit(learner.function, x)] = x

                # Collect and results and add them to the learner
                futures = list(xs.keys())
                done, _ = await asyncio.wait(futures,
                                             return_when=first_completed,
                                             loop=ioloop)
                for fut in done:
                    x = xs.pop(fut)
                    y = await fut
                    learner.add_point(x, y)
        finally:
            learner.remove_unfinished()
            # cancel any outstanding tasks
            cancelled = all(fut.cancel() for fut in xs.keys())
            if not cancelled:
                raise RuntimeError('Some futures remain uncancelled')


# Internal functionality

class _AsyncExecutor:

    def __init__(self, executor, ioloop):
        self.executor = executor
        self.ioloop = ioloop

    def submit(self, f, *args, **kwargs):
        return self.ioloop.run_in_executor(self.executor, f, *args, **kwargs)


def _ensure_async_executor(executor, ioloop):
    if isinstance(executor, ipyparallel.Client):
        executor = executor.executor()
    elif isinstance(executor, (concurrent.ProcessPoolExecutor,
                               concurrent.ThreadPoolExecutor)):
        pass
    elif executor is None:
        executor = concurrent.ProcessPoolExecutor()
    else:
        raise TypeError('Only concurrent.futures.Executors or ipyparallel '
                        'clients can be used.')

    return _AsyncExecutor(executor, ioloop)


def _get_executor_ncores(executor):

    if isinstance(executor, _AsyncExecutor):
        executor = executor.executor

    if isinstance(executor, ipyparallel.client.view.ViewExecutor):
        return len(executor.view)
    elif isinstance(executor, (concurrent.ProcessPoolExecutor,
                               concurrent.ThreadPoolExecutor)):
        return executor._max_workers  # not public API!
    else:
        raise TypeError('Only concurrent.futures.Executors or ipyparallel '
                        'clients can be used.')
