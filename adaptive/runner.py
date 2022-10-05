from __future__ import annotations

import abc
import asyncio
import concurrent.futures as concurrent
import functools
import inspect
import itertools
import pickle
import platform
import sys
import time
import traceback
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable

import loky

from adaptive.notebook_integration import in_ipynb, live_info, live_plot

if TYPE_CHECKING:
    from adaptive import BaseLearner

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
    import mpi4py.futures

    with_mpi4py = True
except ModuleNotFoundError:
    with_mpi4py = False


with suppress(ModuleNotFoundError):
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


if platform.system() == "Linux":
    _default_executor = concurrent.ProcessPoolExecutor
else:
    # On Windows and MacOS functions, the __main__ module must be
    # importable by worker subprocesses. This means that
    # ProcessPoolExecutor will not work in the interactive interpreter.
    # On Linux the whole process is forked, so the issue does not appear.
    # See https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
    # and https://github.com/python-adaptive/adaptive/issues/301
    _default_executor = loky.get_reusable_executor


class BaseRunner(metaclass=abc.ABCMeta):
    r"""Base class for runners that use `concurrent.futures.Executors`.

    Parameters
    ----------
    learner : `~adaptive.BaseLearner` instance
    goal : callable
        The end condition for the calculation. This function must take
        the learner as its sole argument, and return True when we should
        stop requesting more points.
    executor : `concurrent.futures.Executor`, `distributed.Client`,\
               `mpi4py.futures.MPIPoolExecutor`, `ipyparallel.Client` or\
               `loky.get_reusable_executor`, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new `~concurrent.futures.ProcessPoolExecutor` on
        Linux, and a `loky.get_reusable_executor` on MacOS and Windows.
    ntasks : int, optional
        The number of concurrent function evaluations. Defaults to the number
        of cores available in `executor`.
    log : bool, default: False
        If True, record the method calls made to the learner by this runner.
    shutdown_executor : bool, default: False
        If True, shutdown the executor when the runner has completed. If
        `executor` is not provided then the executor created internally
        by the runner is shut down, regardless of this parameter.
    retries : int, default: 0
        Maximum amount of retries of a certain point ``x`` in
        ``learner.function(x)``. After `retries` is reached for ``x``
        the point is present in ``runner.failed``.
    raise_if_retries_exceeded : bool, default: True
        Raise the error after a point ``x`` failed `retries`.

    Attributes
    ----------
    learner : `~adaptive.BaseLearner` instance
        The underlying learner. May be queried for its state.
    log : list or None
        Record of the method calls made to the learner, in the format
        ``(method_name, *args)``.
    to_retry : list of tuples
        List of ``(point, n_fails)``. When a point has failed
        ``runner.retries`` times it is removed but will be present
        in ``runner.tracebacks``.
    tracebacks : list of tuples
        List of of ``(point, tb)`` for points that failed.
    pending_points : list of tuples
        A list of tuples with ``(concurrent.futures.Future, point)``.

    Methods
    -------
    overhead : callable
        The overhead in percent of using Adaptive. Essentially, this is
        ``100 * (1 - total_elapsed_function_time / self.elapsed_time())``.

    """

    def __init__(
        self,
        learner,
        goal,
        *,
        executor=None,
        ntasks=None,
        log=False,
        shutdown_executor=False,
        retries=0,
        raise_if_retries_exceeded=True,
    ):

        self.executor = _ensure_executor(executor)
        self.goal = goal

        self._max_tasks = ntasks

        self._pending_tasks = {}  # mapping from concurrent.futures.Future → point id

        # if we instantiate our own executor, then we are also responsible
        # for calling 'shutdown'
        self.shutdown_executor = shutdown_executor or (executor is None)

        self.learner = learner
        self.log = [] if log else None

        # Timing
        self.start_time = time.time()
        self.end_time = None
        self._elapsed_function_time = 0

        # Error handling attributes
        self.retries = retries
        self.raise_if_retries_exceeded = raise_if_retries_exceeded
        self._to_retry = {}
        self._tracebacks = {}

        self._id_to_point = {}
        self._next_id = functools.partial(
            next, itertools.count()
        )  # some unique id to be associated with each point

    def _get_max_tasks(self):
        return self._max_tasks or _get_ncores(self.executor)

    def _do_raise(self, e, i):
        tb = self._tracebacks[i]
        x = self._id_to_point[i]
        raise RuntimeError(
            "An error occured while evaluating "
            f'"learner.function({x})". '
            f"See the traceback for details.:\n\n{tb}"
        ) from e

    @property
    def do_log(self):
        return self.log is not None

    def _ask(self, n):
        pending_ids = self._pending_tasks.values()
        # using generator here because we only need until `n`
        pids_gen = (pid for pid in self._to_retry.keys() if pid not in pending_ids)
        pids = list(itertools.islice(pids_gen, n))

        loss_improvements = len(pids) * [float("inf")]

        if len(pids) < n:
            new_points, new_losses = self.learner.ask(n - len(pids))
            loss_improvements += new_losses
            for point in new_points:
                pid = self._next_id()
                self._id_to_point[pid] = point
                pids.append(pid)
        return pids, loss_improvements

    def overhead(self):
        """Overhead of using Adaptive and the executor in percent.

        This is measured as ``100 * (1 - t_function / t_elapsed)``.

        Notes
        -----
        This includes the overhead of the executor that is being used.
        The slower your function is, the lower the overhead will be. The
        learners take ~5-50 ms to suggest a point and sending that point to
        the executor also takes about ~5 ms, so you will benefit from using
        Adaptive whenever executing the function takes longer than 100 ms.
        This of course depends on the type of executor and the type of learner
        but is a rough rule of thumb.
        """
        t_function = self._elapsed_function_time
        if t_function == 0:
            # When no function is done executing, the overhead cannot
            # reliably be determined, so 0 is the best we can do.
            return 0
        t_total = self.elapsed_time()
        return (1 - t_function / t_total) * 100

    def _process_futures(self, done_futs):
        for fut in done_futs:
            pid = self._pending_tasks.pop(fut)
            try:
                y = fut.result()
                t = time.time() - fut.start_time  # total execution time
            except Exception as e:
                self._tracebacks[pid] = traceback.format_exc()
                self._to_retry[pid] = self._to_retry.get(pid, 0) + 1
                if self._to_retry[pid] > self.retries:
                    self._to_retry.pop(pid)
                    if self.raise_if_retries_exceeded:
                        self._do_raise(e, pid)
            else:
                self._elapsed_function_time += t / self._get_max_tasks()
                self._to_retry.pop(pid, None)
                self._tracebacks.pop(pid, None)
                x = self._id_to_point.pop(pid)
                if self.do_log:
                    self.log.append(("tell", x, y))
                self.learner.tell(x, y)

    def _get_futures(self):
        # Launch tasks to replace the ones that completed
        # on the last iteration, making sure to fill workers
        # that have started since the last iteration.
        n_new_tasks = max(0, self._get_max_tasks() - len(self._pending_tasks))

        if self.do_log:
            self.log.append(("ask", n_new_tasks))

        pids, _ = self._ask(n_new_tasks)

        for pid in pids:
            start_time = time.time()  # so we can measure execution time
            point = self._id_to_point[pid]
            fut = self._submit(point)
            fut.start_time = start_time
            self._pending_tasks[fut] = pid

        # Collect and results and add them to the learner
        futures = list(self._pending_tasks.keys())
        return futures

    def _remove_unfinished(self):
        # remove points with 'None' values from the learner
        self.learner.remove_unfinished()
        # cancel any outstanding tasks
        remaining = list(self._pending_tasks.keys())
        for fut in remaining:
            fut.cancel()
        return remaining

    def _cleanup(self):
        if self.shutdown_executor:
            # XXX: temporary set wait=True because of a bug with Python ≥3.7
            # and loky in any Python version.
            # see https://github.com/python-adaptive/adaptive/issues/156
            # and https://github.com/python-adaptive/adaptive/pull/164
            # and https://bugs.python.org/issue36281
            # and https://github.com/joblib/loky/issues/241
            self.executor.shutdown(wait=True)
        self.end_time = time.time()

    @property
    def failed(self):
        """Set of points that failed ``runner.retries`` times."""
        return set(self._tracebacks) - set(self._to_retry)

    @abc.abstractmethod
    def elapsed_time(self):
        """Return the total time elapsed since the runner
        was started.

        Is called in `overhead`.
        """

    @abc.abstractmethod
    def _submit(self, x):
        """Is called in `_get_futures`."""

    @property
    def tracebacks(self):
        return [(self._id_to_point[pid], tb) for pid, tb in self._tracebacks.items()]

    @property
    def to_retry(self):
        return [(self._id_to_point[pid], n) for pid, n in self._to_retry.items()]

    @property
    def pending_points(self):
        return [
            (fut, self._id_to_point[pid]) for fut, pid in self._pending_tasks.items()
        ]


class BlockingRunner(BaseRunner):
    """Run a learner synchronously in an executor.

    Parameters
    ----------
    learner : `~adaptive.BaseLearner` instance
    goal : callable
        The end condition for the calculation. This function must take
        the learner as its sole argument, and return True when we should
        stop requesting more points.
    executor : `concurrent.futures.Executor`, `distributed.Client`,\
               `mpi4py.futures.MPIPoolExecutor`, `ipyparallel.Client` or\
               `loky.get_reusable_executor`, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new `~concurrent.futures.ProcessPoolExecutor` on
        Linux, and a `loky.get_reusable_executor` on MacOS and Windows.
    ntasks : int, optional
        The number of concurrent function evaluations. Defaults to the number
        of cores available in `executor`.
    log : bool, default: False
        If True, record the method calls made to the learner by this runner.
    shutdown_executor : bool, default: False
        If True, shutdown the executor when the runner has completed. If
        `executor` is not provided then the executor created internally
        by the runner is shut down, regardless of this parameter.
    retries : int, default: 0
        Maximum amount of retries of a certain point ``x`` in
        ``learner.function(x)``. After `retries` is reached for ``x``
        the point is present in ``runner.failed``.
    raise_if_retries_exceeded : bool, default: True
        Raise the error after a point ``x`` failed `retries`.

    Attributes
    ----------
    learner : `~adaptive.BaseLearner` instance
        The underlying learner. May be queried for its state.
    log : list or None
        Record of the method calls made to the learner, in the format
        ``(method_name, *args)``.
    to_retry : list of tuples
        List of ``(point, n_fails)``. When a point has failed
        ``runner.retries`` times it is removed but will be present
        in ``runner.tracebacks``.
    tracebacks : list of tuples
        List of of ``(point, tb)`` for points that failed.
    pending_points : list of tuples
        A list of tuples with ``(concurrent.futures.Future, point)``.

    Methods
    -------
    elapsed_time : callable
        A method that returns the time elapsed since the runner
        was started.
    overhead : callable
        The overhead in percent of using Adaptive. This includes the
        overhead of the executor. Essentially, this is
        ``100 * (1 - total_elapsed_function_time / self.elapsed_time())``.

    """

    def __init__(
        self,
        learner,
        goal,
        *,
        executor=None,
        ntasks=None,
        log=False,
        shutdown_executor=False,
        retries=0,
        raise_if_retries_exceeded=True,
    ):
        if inspect.iscoroutinefunction(learner.function):
            raise ValueError("Coroutine functions can only be used with 'AsyncRunner'.")
        super().__init__(
            learner,
            goal,
            executor=executor,
            ntasks=ntasks,
            log=log,
            shutdown_executor=shutdown_executor,
            retries=retries,
            raise_if_retries_exceeded=raise_if_retries_exceeded,
        )
        self._run()

    def _submit(self, x):
        return self.executor.submit(self.learner.function, x)

    def _run(self):
        first_completed = concurrent.FIRST_COMPLETED

        if self._get_max_tasks() < 1:
            raise RuntimeError("Executor has no workers")

        try:
            while not self.goal(self.learner):
                futures = self._get_futures()
                done, _ = concurrent.wait(futures, return_when=first_completed)
                self._process_futures(done)
        finally:
            remaining = self._remove_unfinished()
            if remaining:
                concurrent.wait(remaining)
                # Some futures get their result set, despite being cancelled.
                # see https://github.com/python-adaptive/adaptive/issues/319
                with_result = [f for f in remaining if not f.cancelled() and f.done()]
                self._process_futures(with_result)
            self._cleanup()

    def elapsed_time(self):
        """Return the total time elapsed since the runner
        was started."""
        if self.end_time is None:
            # This shouldn't happen if the BlockingRunner
            # correctly finished.
            self.end_time = time.time()
        return self.end_time - self.start_time


class AsyncRunner(BaseRunner):
    r"""Run a learner asynchronously in an executor using `asyncio`.

    Parameters
    ----------
    learner : `~adaptive.BaseLearner` instance
    goal : callable, optional
        The end condition for the calculation. This function must take
        the learner as its sole argument, and return True when we should
        stop requesting more points. If not provided, the runner will run
        forever, or until ``self.task.cancel()`` is called.
    executor : `concurrent.futures.Executor`, `distributed.Client`,\
               `mpi4py.futures.MPIPoolExecutor`, `ipyparallel.Client` or\
               `loky.get_reusable_executor`, optional
        The executor in which to evaluate the function to be learned.
        If not provided, a new `~concurrent.futures.ProcessPoolExecutor` on
        Linux, and a `loky.get_reusable_executor` on MacOS and Windows.
    ntasks : int, optional
        The number of concurrent function evaluations. Defaults to the number
        of cores available in `executor`.
    log : bool, default: False
        If True, record the method calls made to the learner by this runner.
    shutdown_executor : bool, default: False
        If True, shutdown the executor when the runner has completed. If
        `executor` is not provided then the executor created internally
        by the runner is shut down, regardless of this parameter.
    ioloop : ``asyncio.AbstractEventLoop``, optional
        The ioloop in which to run the learning algorithm. If not provided,
        the default event loop is used.
    retries : int, default: 0
        Maximum amount of retries of a certain point ``x`` in
        ``learner.function(x)``. After `retries` is reached for ``x``
        the point is present in ``runner.failed``.
    raise_if_retries_exceeded : bool, default: True
        Raise the error after a point ``x`` failed `retries`.

    Attributes
    ----------
    task : `asyncio.Task`
        The underlying task. May be cancelled in order to stop the runner.
    learner : `~adaptive.BaseLearner` instance
        The underlying learner. May be queried for its state.
    log : list or None
        Record of the method calls made to the learner, in the format
        ``(method_name, *args)``.
    to_retry : list of tuples
        List of ``(point, n_fails)``. When a point has failed
        ``runner.retries`` times it is removed but will be present
        in ``runner.tracebacks``.
    tracebacks : list of tuples
        List of of ``(point, tb)`` for points that failed.
    pending_points : list of tuples
        A list of tuples with ``(concurrent.futures.Future, point)``.

    Methods
    -------
    elapsed_time : callable
        A method that returns the time elapsed since the runner
        was started.
    overhead : callable
        The overhead in percent of using Adaptive. This includes the
        overhead of the executor. Essentially, this is
        ``100 * (1 - total_elapsed_function_time / self.elapsed_time())``.

    Notes
    -----
    This runner can be used when an async function (defined with
    ``async def``) has to be learned. In this case the function will be
    run directly on the event loop (and not in the executor).
    """

    def __init__(
        self,
        learner,
        goal=None,
        *,
        executor=None,
        ntasks=None,
        log=False,
        shutdown_executor=False,
        ioloop=None,
        retries=0,
        raise_if_retries_exceeded=True,
    ):

        if goal is None:

            def goal(_):
                return False

        if (
            executor is None
            and _default_executor is concurrent.ProcessPoolExecutor
            and not inspect.iscoroutinefunction(learner.function)
        ):
            try:
                pickle.dumps(learner.function)
            except pickle.PicklingError:
                raise ValueError(
                    "`learner.function` cannot be pickled (is it a lamdba function?)"
                    " and therefore does not work with the default executor."
                    " Either make sure the function is pickleble or use an executor"
                    " that might work with 'hard to pickle'-functions"
                    " , e.g. `ipyparallel` with `dill`."
                )

        super().__init__(
            learner,
            goal,
            executor=executor,
            ntasks=ntasks,
            log=log,
            shutdown_executor=shutdown_executor,
            retries=retries,
            raise_if_retries_exceeded=raise_if_retries_exceeded,
        )
        self.ioloop = ioloop or asyncio.get_event_loop()
        self.task = None

        # When the learned function is 'async def', we run it
        # directly on the event loop, and not in the executor.
        # The *whole point* of allowing learning of async functions is so that
        # the user can have more fine-grained control over the parallelism.
        if inspect.iscoroutinefunction(learner.function):
            if executor:  # user-provided argument
                raise RuntimeError(
                    "Cannot use an executor when learning an async function."
                )
            self.executor.shutdown()  # Make sure we don't shoot ourselves later

        self.task = self.ioloop.create_task(self._run())
        self.saving_task = None
        if in_ipynb() and not self.ioloop.is_running():
            warnings.warn(
                "The runner has been scheduled, but the asyncio "
                "event loop is not running! If you are "
                "in a Jupyter notebook, remember to run "
                "'adaptive.notebook_extension()'"
            )

    def _submit(self, x):
        ioloop = self.ioloop
        if inspect.iscoroutinefunction(self.learner.function):
            return ioloop.create_task(self.learner.function(x))
        else:
            return ioloop.run_in_executor(self.executor, self.learner.function, x)

    def status(self):
        """Return the runner status as a string.

        The possible statuses are: running, cancelled, failed, and finished.
        """
        try:
            self.task.result()
        except asyncio.CancelledError:
            return "cancelled"
        except asyncio.InvalidStateError:
            return "running"
        except Exception:
            return "failed"
        else:
            return "finished"

    def cancel(self):
        """Cancel the runner.

        This is equivalent to calling ``runner.task.cancel()``.
        """
        self.task.cancel()

    def live_plot(self, *, plotter=None, update_interval=2, name=None, normalize=True):
        """Live plotting of the learner's data.

        Parameters
        ----------
        runner : `~adaptive.Runner`
        plotter : function
            A function that takes the learner as a argument and returns a
            holoviews object. By default ``learner.plot()`` will be called.
        update_interval : int
            Number of second between the updates of the plot.
        name : hasable
            Name for the `live_plot` task in `adaptive.active_plotting_tasks`.
            By default the name is None and if another task with the same name
            already exists that other `live_plot` is canceled.
        normalize : bool
            Normalize (scale to fit) the frame upon each update.

        Returns
        -------
        dm : `holoviews.core.DynamicMap`
            The plot that automatically updates every `update_interval`.
        """
        return live_plot(
            self, plotter=plotter, update_interval=update_interval, name=name
        )

    def live_info(self, *, update_interval=0.1):
        """Display live information about the runner.

        Returns an interactive ipywidget that can be
        visualized in a Jupyter notebook.
        """
        return live_info(self, update_interval=update_interval)

    async def _run(self):
        first_completed = asyncio.FIRST_COMPLETED

        if self._get_max_tasks() < 1:
            raise RuntimeError("Executor has no workers")

        try:
            while not self.goal(self.learner):
                futures = self._get_futures()
                kw = {"loop": self.ioloop} if sys.version_info[:2] < (3, 10) else {}
                done, _ = await asyncio.wait(futures, return_when=first_completed, **kw)
                self._process_futures(done)
        finally:
            remaining = self._remove_unfinished()
            if remaining:
                await asyncio.wait(remaining)
            self._cleanup()

    def elapsed_time(self):
        """Return the total time elapsed since the runner
        was started."""
        if self.task.done():
            end_time = self.end_time
            if end_time is None:
                # task was cancelled before it began
                assert self.task.cancelled()
                return 0
        else:
            end_time = time.time()
        return end_time - self.start_time

    def start_periodic_saving(
        self,
        save_kwargs: dict[str, Any] | None = None,
        interval: int = 30,
        method: Callable[[BaseLearner], None] | None = None,
    ):
        """Periodically save the learner's data.

        Parameters
        ----------
        save_kwargs : dict
            Key-word arguments for ``learner.save(**save_kwargs)``.
            Only used if ``method=None``.
        interval : int
            Number of seconds between saving the learner.
        method : callable
            The method to use for saving the learner. If None, the default
            saves the learner using "pickle" which calls
            ``learner.save(**save_kwargs)``. Otherwise provide a callable
            that takes the learner and saves the learner.

        Example
        -------
        >>> runner = Runner(learner)
        >>> runner.start_periodic_saving(
        ...     save_kwargs=dict(fname='data/test.pickle'),
        ...     interval=600)
        """

        def default_save(learner):
            learner.save(**save_kwargs)

        if method is None:
            method = default_save
            if save_kwargs is None:
                raise ValueError("Must provide `save_kwargs` if method=None.")

        async def _saver():
            while self.status() == "running":
                method(self.learner)
                await asyncio.sleep(interval)
            method(self.learner)  # one last time

        self.saving_task = self.ioloop.create_task(_saver())
        return self.saving_task


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
    learner : ~`adaptive.BaseLearner` instance
    goal : callable
        The end condition for the calculation. This function must take the
        learner as its sole argument, and return True if we should stop.
    """
    while not goal(learner):
        xs, _ = learner.ask(1)
        for x in xs:
            y = learner.function(x)
            learner.tell(x, y)


def replay_log(learner, log):
    """Apply a sequence of method calls to a learner.

    This is useful for debugging runners.

    Parameters
    ----------
    learner : `~adaptive.BaseLearner` instance
        New learner where the log will be applied.
    log : list
        contains tuples: ``(method_name, *args)``.
    """
    for method, *args in log:
        getattr(learner, method)(*args)


# --- Useful runner goals


def stop_after(*, seconds=0, minutes=0, hours=0):
    """Stop a runner after a specified time.

    For example, to specify a runner that should stop after
    5 minutes, one could do the following:

    >>> runner = Runner(learner, goal=stop_after(minutes=5))

    To stop a runner after 2 hours, 10 minutes and 3 seconds,
    one could do the following:

    >>> runner = Runner(learner, goal=stop_after(hours=2, minutes=10, seconds=3))

    Parameters
    ----------
    seconds, minutes, hours : float, default: 0
        If more than one is specified, then they are added together

    Returns
    -------
    goal : callable
        Can be used as the ``goal`` parameter when constructing
        a `Runner`.

    Notes
    -----
    The duration specified is only a *lower bound* on the time that the
    runner will run for, because the runner only checks its goal when
    it adds points to its learner
    """
    stop_time = time.time() + seconds + 60 * minutes + 3600 * hours
    return lambda _: time.time() > stop_time


# -- Internal executor-related, things


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
        executor = _default_executor()

    if isinstance(executor, concurrent.Executor):
        return executor
    elif with_ipyparallel and isinstance(executor, ipyparallel.Client):
        return executor.executor()
    elif with_distributed and isinstance(
        executor, (distributed.Client, distributed.client.Client)
    ):
        return executor.get_executor()
    else:
        raise TypeError(
            "Only a concurrent.futures.Executor, distributed.Client,"
            " or ipyparallel.Client can be used."
        )


def _get_ncores(ex):
    """Return the maximum  number of cores that an executor can use."""
    if with_ipyparallel and isinstance(ex, ipyparallel.client.view.ViewExecutor):
        return len(ex.view)
    elif isinstance(
        ex, (concurrent.ProcessPoolExecutor, concurrent.ThreadPoolExecutor)
    ):
        return ex._max_workers  # not public API!
    elif isinstance(ex, loky.reusable_executor._ReusablePoolExecutor):
        return ex._max_workers  # not public API!
    elif isinstance(ex, SequentialExecutor):
        return 1
    elif with_distributed and isinstance(ex, distributed.cfexecutor.ClientExecutor):
        return sum(n for n in ex._client.ncores().values())
    elif with_mpi4py and isinstance(ex, mpi4py.futures.MPIPoolExecutor):
        ex.bootup()  # wait until all workers are up and running
        return ex._pool.size  # not public API!
    else:
        raise TypeError(f"Cannot get number of cores for {ex.__class__}")
