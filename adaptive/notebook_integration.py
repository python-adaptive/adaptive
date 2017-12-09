# -*- coding: utf-8 -*-
import asyncio
from pkg_resources import parse_version
import warnings

from IPython import get_ipython
import ipykernel
from ipykernel.eventloops import register_integration


# IPython event loop integration

if parse_version(ipykernel.__version__) < parse_version('4.7.0'):
    # XXX: remove this function when we depend on ipykernel>=4.7.0
    @register_integration('asyncio')
    def _loop_asyncio(kernel):
        """Start a kernel with asyncio event loop support.
        Taken from https://github.com/ipython/ipykernel/blob/fa814da201bdebd5b16110597604f7dabafec58d/ipykernel/eventloops.py#L294"""
        loop = asyncio.get_event_loop()

        def kernel_handler():
            loop.call_soon(kernel.do_one_iteration)
            loop.call_later(kernel._poll_interval, kernel_handler)

        loop.call_soon(kernel_handler)
        # loop is already running (e.g. tornado 5), nothing left to do
        if loop.is_running():
            return
        while True:
            error = None
            try:
                loop.run_forever()
            except KeyboardInterrupt:
                continue
            except Exception as e:
                error = e
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            if error is not None:
                raise error
            break


def notebook_extension():
    get_ipython().magic('gui asyncio')
    try:
        import holoviews as hv
        return hv.notebook_extension('bokeh')
    except ModuleNotFoundError:
        warnings.warn("The holoviews package is not installed so plotting"
                      "will not work.", RuntimeWarning)


# Plotting

active_plotting_tasks = dict()

# Incremented by 'live_plot' on every successful plot creation;
# used to name plots that are not given an explicit name.
_last_plot_id = 0


def live_plot(runner, *, plotter=None, update_interval=2, name=None):
    try:
        import holoviews as hv
    except ModuleNotFoundError:
        raise RuntimeError('Plotting requires the holoviews Python package'
                           ' which is not installed.')

    def plot_generator():
        while True:
            if not plotter:
                yield runner.learner.plot()
            else:
                yield plotter(runner.learner)

    dm = hv.DynamicMap(plot_generator(),
                       streams=[hv.streams.Stream.define('Next')()])

    # Generate task name if not provided
    global _last_plot_id
    if not name:
        name = f'plot_{_last_plot_id}'
    _last_plot_id += 1

    # Could have used dm.periodic in the following, but this would either spin
    # off a thread (and learner is not threadsafe) or block the kernel.

    async def updater():
        try:
            while not runner.task.done():
                dm.event()
                await asyncio.sleep(update_interval)
            dm.event()  # fire off one last update before we die
        finally:
            active_plotting_tasks.pop(name, None)

    task = asyncio.get_event_loop().create_task(updater())

    active_plotting_tasks[name] = task

    return dm
