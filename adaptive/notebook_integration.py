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


def live_plot(runner, *, plotter=None, update_interval=2, name=None):
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
    try:
        import holoviews as hv
    except ModuleNotFoundError:
        raise RuntimeError('Plotting requires the holoviews Python package'
                           ' which is not installed.')
    import ipywidgets
    from IPython.display import display

    def plot_generator():
        while True:
            if not plotter:
                yield runner.learner.plot()
            else:
                yield plotter(runner.learner)

    dm = hv.DynamicMap(plot_generator(),
                       streams=[hv.streams.Stream.define('Next')()])


    # Could have used dm.periodic in the following, but this would either spin
    # off a thread (and learner is not threadsafe) or block the kernel.

    async def updater():
        try:
            while not runner.task.done():
                dm.event()
                await asyncio.sleep(update_interval)
            dm.event()  # fire off one last update before we die
        finally:
            if active_plotting_tasks[name] is asyncio.Task.current_task():
                active_plotting_tasks.pop(name, None)

    global active_plotting_tasks
    if name in active_plotting_tasks:
        active_plotting_tasks[name].cancel()

    active_plotting_tasks[name] = asyncio.get_event_loop().create_task(updater())

    def cancel(_):
        try:
            active_plotting_tasks[name].cancel()
        except KeyError:
            pass

    cancel_button = ipywidgets.Button(description='cancel live-plot',
                                      layout=ipywidgets.Layout(width='150px'))
    cancel_button.on_click(cancel)
    display(cancel_button)

    return dm
