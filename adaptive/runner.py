# -*- coding: utf-8 -*-
import asyncio
import concurrent.futures as concurrent
import functools
import inspect
import os
import time
import traceback
import warnings

from .notebook_integration import live_plot, live_info, in_ipynb
from .utils import WithTime, AverageTimeReturn

try:
    import ipyparallel
    with_ipyparallel = True
except ModuleNotFoundError:
    with_ipyparallel = False

try:
    import distributed
    with_distributed = True
except ModuleNotFoundError:
    with_distributed = False

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ModuleNotFoundError:
    pass


if os.name == 'nt':
    if with_distributed:
        _default_executor = distributed.Client
        _default_executor_kwargs = {'address': distributed.LocalCluster()}
    else:
        _windows_executor_msg = (
            "The default executor on Windows for 'adaptive.Runner' cannot "
            "be used because the package 'distributed' is not installed. "
            "Either install 'distributed' or explicitly specify an executor "
            "when using 'adaptive.Runner'."
        )

        _default_executor_kwargs = {}

        def _default_executor(*args, **kwargs):
            raise RuntimeError(_windows_executor_msg)

        warnings.warn(_windows_executor_msg)

else:
    _default_executor = concurrent.ProcessPoolExecutor
    _default_executor_kwargs = {}



class BaseRunner:
    """Base class for runners that use concurrent.futures.Executors.

    Parameters
    ----------
    learner : adaptive.learner.BaseLearner
    goal : callable
        The end condition for the calculation. This function must take
        the learner as its sole argument, and return True when we should
        stop requesting more points.
    executor : concurrent.futures.Executor, or ipyparallel.Client, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new ProcessPoolExecutor is used.
    ntasks : int, optional
        The number of concurrent function evaluations. Defaults to the number
        of cores available in 'executor'.
    log : bool, default: False
        If True, record the method calls made to the learner by this runner
    shutdown_executor : Bool, default: False
        If True, shutdown the executor when the runner has completed. If
        'executor' is not provided then the executor created internally
        by the runner is shut down, regardless of this parameter.

    Attributes
    ----------
    learner : Learner
        The underlying learner. May be queried for its state
    log : list or None
        Record of the method calls made to the learner, in the format
        '(method_name, *args)'.
    """

    def __init__(self, learner, goal, *,
                 executor=None, ntasks=None, log=False,
                 shutdown_executor=False):

        self.executor = _ensure_executor(executor)
        self.goal = goal

        self._max_tasks = ntasks
            
        # if we instantiate our own executor, then we are also responsible
        # for calling 'shutdown'
        self.shutdown_executor = shutdown_executor or (executor is None)

        self.learner = learner
        self.log = [] if log else None
        self.task = None

    def max_tasks(self):
        return self._max_tasks or _get_ncores(self.executor)


class BlockingRunner(BaseRunner):
    """Run a learner synchronously in an executor.

    Parameters
    ----------
    learner : adaptive.learner.BaseLearner
    goal : callable
        The end condition for the calculation. This function must take
        the learner as its sole argument, and return True when we should
        stop requesting more points.
    executor : concurrent.futures.Executor, distributed.Client,
               or ipyparallel.Client, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new `ProcessPoolExecutor` is used on Unix systems
        while on Windows a `distributed.Client` is used if `distributed` is
        installed.
    ntasks : int, optional
        The number of concurrent function evaluations. Defaults to the number
        of cores available in 'executor'.
    log : bool, default: False
        If True, record the method calls made to the learner by this runner
    shutdown_executor : Bool, default: False
        If True, shutdown the executor when the runner has completed. If
        'executor' is not provided then the executor created internally
        by the runner is shut down, regardless of this parameter.

    Attributes
    ----------
    learner : Learner
        The underlying learner. May be queried for its state
    log : list or None
        Record of the method calls made to the learner, in the format
        '(method_name, *args)'.
    """

    def __init__(self, learner, goal, *,
                 executor=None, ntasks=None, log=False,
                 shutdown_executor=False):
        if inspect.iscoroutinefunction(learner.function):
            raise ValueError("Coroutine functions can only be used "
                             "with 'AsyncRunner'.")
        super().__init__(learner, goal, executor=executor, ntasks=ntasks,
                         log=log, shutdown_executor=shutdown_executor)
        self._run()

    def _submit(self, x):
        return self.executor.submit(self.learner.function, x)

    def _run(self):
        first_completed = concurrent.FIRST_COMPLETED
        xs = dict()
        do_log = self.log is not None

        if self.max_tasks() < 1:
            raise RuntimeError('Executor has no workers')

        try:
            while not self.goal(self.learner):
                # Launch tasks to replace the ones that completed
                # on the last iteration, making sure to fill workers
                # that have started since the last iteration.
                n_new_tasks = max(0, self.max_tasks() - len(xs))

                if do_log:
                    self.log.append(('ask', n_new_tasks))

                points, _ = self.learner.ask(n_new_tasks)

                for x in points:
                    xs[self._submit(x)] = x

                # Collect and results and add them to the learner
                futures = list(xs.keys())
                done, _ = concurrent.wait(futures,
                                          return_when=first_completed)
                for fut in done:
                    x = xs.pop(fut)
                    try:
                        y = fut.result()
                    except Exception as e:
                        tb = traceback.format_exc()
                        raise RuntimeError(
                            'An error occured while evaluating '
                            f'"learner.function({x})". '
                            f'See the traceback for details.:\n\n{tb}'
                        ) from e
                    if do_log:
                        self.log.append(('tell', x, y))
                    self.learner._tell(x, y)

        finally:
            # remove points with 'None' values from the learner
            self.learner.remove_unfinished()
            # cancel any outstanding tasks
            remaining = list(xs.keys())
            if remaining:
                for fut in remaining:
                    fut.cancel()
                concurrent.wait(remaining)
            if self.shutdown_executor:
                self.executor.shutdown()


class AsyncRunner(BaseRunner):
    """Run a learner asynchronously in an executor using asyncio.

    Parameters
    ----------
    learner : adaptive.learner.BaseLearner
    goal : callable, optional
        The end condition for the calculation. This function must take
        the learner as its sole argument, and return True when we should
        stop requesting more points. If not provided, the runner will run
        forever, or until 'self.task.cancel()' is called.
    executor : concurrent.futures.Executor, distributed.Client,
               or ipyparallel.Client, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new `ProcessPoolExecutor` is used on Unix systems
        while on Windows a `distributed.Client` is used if `distributed` is
        installed.
    ntasks : int, optional
        The number of concurrent function evaluations. Defaults to the number
        of cores available in 'executor'.
    log : bool, default: False
        If True, record the method calls made to the learner by this runner
    shutdown_executor : Bool, default: False
        If True, shutdown the executor when the runner has completed. If
        'executor' is not provided then the executor created internally
        by the runner is shut down, regardless of this parameter.
    ioloop : asyncio.AbstractEventLoop, optional
        The ioloop in which to run the learning algorithm. If not provided,
        the default event loop is used.

    Attributes
    ----------
    task : asyncio.Task
        The underlying task. May be cancelled in order to stop the runner.
    learner : Learner
        The underlying learner. May be queried for its state.
    log : list or None
        Record of the method calls made to the learner, in the format
        '(method_name, *args)'.

    Notes
    -----
    This runner can be used when an async function (defined with
    'async def') has to be learned. In this case the function will be
    run directly on the event loop (and not in the executor).
    """

    def __init__(self, learner, goal=None, *,
                 executor=None, ntasks=None, log=False,
                 ioloop=None, shutdown_executor=False):

        if goal is None:
            def goal(_):
                return False

        super().__init__(learner, goal, executor=executor, ntasks=ntasks,
                         log=log, shutdown_executor=shutdown_executor)
        self.ioloop = ioloop or asyncio.get_event_loop()
        self.task = None

        self.start_time = time.time()
        self.end_time = None
        self.time_function = 0
        self._npoints = 0

        self._tell = WithTime(self.learner._tell)
        self.ask = WithTime(self.learner.ask)
        self.function = AverageTimeReturn(self.learner.function)

        # When the learned function is 'async def', we run it
        # directly on the event loop, and not in the executor.
        # The *whole point* of allowing learning of async functions is so that
        # the user can have more fine-grained control over the parallelism.
        if inspect.iscoroutinefunction(learner.function):
            if executor:  # user-provided argument
                raise RuntimeError('Cannot use an executor when learning an '
                                   'async function.')
            self.executor.shutdown()  # Make sure we don't shoot ourselves later
            self._submit = lambda x: self.ioloop.create_task(self.function(x))
        else:
            self._submit = functools.partial(self.ioloop.run_in_executor,
                                             self.executor,
                                             self.function)

        self.task = self.ioloop.create_task(self._run())
        if in_ipynb() and not self.ioloop.is_running():
            warnings.warn("The runner has been scheduled, but the asyncio "
                          "event loop is not running! If you are "
                          "in a Jupyter notebook, remember to run "
                          "'adaptive.notebook_extension()'")

    def elapsed_time(self):
        if self.task.done():
            end_time = self.end_time
            if end_time is None:
                # task was cancelled before it began
                assert self.task.cancelled()
                return 0
        else:
            end_time = time.time()
        return end_time - self.start_time

    def performance(self):
        try:
            ncores = _get_ncores(self.executor)
            t_function = self.time_function / ncores
            t_adaptive = (self.ask.time + self._tell.time) / self._npoints
            return t_function / t_adaptive
        except ZeroDivisionError:
            return 42

    def status(self):
        """Return the runner status as a string.

        The possible statuses are: running, cancelled, failed, and finished.
        """
        try:
            self.task.result()
        except asyncio.CancelledError:
            return 'cancelled'
        except asyncio.InvalidStateError:
            return 'running'
        except Exception:
            return 'failed'
        else:
            return 'finished'

    def cancel(self):
        """Cancel the runner.

        This is equivalent to calling `runner.task.cancel()`.
        """
        self.task.cancel()

    def live_plot(self, *, plotter=None, update_interval=2, name=None):
        """Live plotting of the learner's data.

        Parameters
        ----------
        runner : Runner
        plotter : function
            A function that takes the learner as a argument and returns a
            holoviews object. By default learner.plot() will be called.
        update_interval : int
            Number of second between the updates of the plot.
        name : hasable
            Name for the `live_plot` task in `adaptive.active_plotting_tasks`.
            By default the name is `None` and if another task with the same name
            already exists that other live_plot is canceled.

        Returns
        -------
        dm : holoviews.DynamicMap
            The plot that automatically updates every update_interval.
        """
        return live_plot(self, plotter=plotter,
                         update_interval=update_interval,
                         name=name)

    def live_info(self, *, update_interval=0.1):
        """Display live information about the runner.

        Returns an interactive ipywidget that can be
        visualized in a Jupyter notebook.
        """
        return live_info(self, update_interval=update_interval)


    async def _run(self):
        first_completed = asyncio.FIRST_COMPLETED
        xs = dict()  # The points we are waiting for
        do_log = self.log is not None

        if self.max_tasks() < 1:
            raise RuntimeError('Executor has no workers')

        try:
            while not self.goal(self.learner):
                # Launch tasks to replace the ones that completed
                # on the last iteration, making sure to fill workers
                # that have started since the last iteration.
                n_new_tasks = max(0, self.max_tasks() - len(xs))

                if do_log:
                    self.log.append(('ask', n_new_tasks))

                points, _ = self.ask(n_new_tasks)
                for x in points:
                    xs[self._submit(x)] = x

                # Collect and results and add them to the learner
                futures = list(xs.keys())
                done, _ = await asyncio.wait(futures,
                                             return_when=first_completed,
                                             loop=self.ioloop)
                for fut in done:
                    x = xs.pop(fut)
                    try:
                        y, t = fut.result()
                        self.time_function = t
                        self._npoints += 1
                    except Exception as e:
                        tb = traceback.format_exc()
                        raise RuntimeError(
                            'An error occured while evaluating '
                            f'"learner.function({x})". '
                            f'See the traceback for details.:\n\n{tb}'
                        ) from e
                    if do_log:
                        self.log.append(('tell', x, y))
                    self._tell(x, y)
        finally:
            # remove points with 'None' values from the learner
            self.learner.remove_unfinished()
            # cancel any outstanding tasks
            remaining = list(xs.keys())
            if remaining:
                for fut in remaining:
                    fut.cancel()
                await asyncio.wait(remaining)
            if self.shutdown_executor:
                self.executor.shutdown(wait=False)
            self.end_time = time.time()


# Default runner
Runner = AsyncRunner


def simple(learner, goal):
    """Run the learner until the goal is reached.

    Requests a single point from the learner, evaluates
    the function to be learned, and adds the point to the
    learner, until the goal is reached, blocking the current
    thread.

    This function is useful for extracting error messages,
    as the learner's function is evaluated in the same thread,
    meaning that exceptions can simple be caught an inspected.

    Parameters
    ----------
    learner : adaptive.BaseLearner
    goal : callable
        The end condition for the calculation. This function must take the
        learner as its sole argument, and return True if we should stop.
    """
    while not goal(learner):
        xs, _ = learner.ask(1)
        for x in xs:
            y = learner.function(x)
            learner._tell(x, y)


def replay_log(learner, log):
    """Apply a sequence of method calls to a learner.

    This is useful for debugging runners.

    Parameters
    ----------
    learner : learner.BaseLearner
    log : list
        contains tuples: '(method_name, *args)'.
    """
    for method, *args in log:
        getattr(learner, method)(*args)


class SequentialExecutor(concurrent.Executor):
    """A trivial executor that runs functions synchronously.

    This executor is mainly for testing.
    """
    def submit(self, fn, *args, **kwargs):
        fut = concurrent.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:
            fut.set_exception(e)
        return fut

    def map(self, fn, *iterable, timeout=None, chunksize=1):
        return map(fn, iterable)

    def shutdown(self, wait=True):
        pass


def _ensure_executor(executor):
    if executor is None:
        executor = _default_executor(**_default_executor_kwargs)

    if isinstance(executor, concurrent.Executor):
        return executor
    elif with_ipyparallel and isinstance(executor, ipyparallel.Client):
        return executor.executor()
    elif with_distributed and isinstance(executor, distributed.Client):
        return executor.get_executor()
    else:
        raise TypeError('Only a concurrent.futures.Executor, distributed.Client,'
                        ' or ipyparallel.Client can be used.')


def _get_ncores(ex):
    """Return the maximum  number of cores that an executor can use."""
    if with_ipyparallel and isinstance(ex, ipyparallel.client.view.ViewExecutor):
        return len(ex.view)
    elif isinstance(ex, (concurrent.ProcessPoolExecutor,
                         concurrent.ThreadPoolExecutor)):
        return ex._max_workers  # not public API!
    elif isinstance(ex, SequentialExecutor):
        return 1
    elif with_distributed and isinstance(ex, distributed.cfexecutor.ClientExecutor):
        return sum(n for n in ex._client.ncores().values())
    else:
        raise TypeError('Cannot get number of cores for {}'
                        .format(ex.__class__))
