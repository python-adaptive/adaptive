from __future__ import annotations

import asyncio
import datetime
import importlib
import random
import warnings
from contextlib import suppress

_async_enabled = False
_holoviews_enabled = False
_ipywidgets_enabled = False
_plotly_enabled = False


def notebook_extension(*, _inline_js=True):
    """Enable ipywidgets, holoviews, and asyncio notebook integration."""
    if not in_ipynb():
        raise RuntimeError(
            '"adaptive.notebook_extension()" may only be run '
            "from a Jupyter notebook."
        )

    global _async_enabled, _holoviews_enabled, _ipywidgets_enabled

    # Load holoviews
    try:
        _holoviews_enabled = False  # After closing a notebook the js is gone
        if not _holoviews_enabled:
            import holoviews

            holoviews.notebook_extension("bokeh", logo=False, inline=_inline_js)
            _holoviews_enabled = True
    except ModuleNotFoundError:
        warnings.warn(
            "holoviews is not installed; plotting is disabled.", RuntimeWarning
        )

    # Load ipywidgets
    try:
        if not _ipywidgets_enabled:
            import ipywidgets  # noqa: F401

            _ipywidgets_enabled = True
    except ModuleNotFoundError:
        warnings.warn(
            "ipywidgets is not installed; live_info is disabled.", RuntimeWarning
        )

    # Enable asyncio integration
    if not _async_enabled:
        get_ipython().magic("gui asyncio")  # noqa: F821
        _async_enabled = True


def ensure_holoviews():
    try:
        return importlib.import_module("holoviews")
    except ModuleNotFoundError:
        raise RuntimeError("holoviews is not installed; plotting is disabled.")


def ensure_plotly():
    global _plotly_enabled
    try:
        import plotly

        if not _plotly_enabled:
            import plotly.figure_factory
            import plotly.graph_objs
            import plotly.offline

            # This injects javascript and should happen only once
            plotly.offline.init_notebook_mode()
            _plotly_enabled = True
        return plotly
    except ModuleNotFoundError:
        raise RuntimeError("plotly is not installed; plotting is disabled.")


def in_ipynb() -> bool:
    try:
        # If we are running in IPython, then `get_ipython()` is always a global
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


# Fancy displays in the Jupyter notebook

active_plotting_tasks = dict()


def live_plot(runner, *, plotter=None, update_interval=2, name=None, normalize=True):
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
    if not _holoviews_enabled:
        raise RuntimeError(
            "Live plotting is not enabled; did you run "
            "'adaptive.notebook_extension()'?"
        )

    import holoviews as hv
    import ipywidgets
    from IPython.display import display

    if name in active_plotting_tasks:
        active_plotting_tasks[name].cancel()

    def plot_generator():
        while True:
            if not plotter:
                yield runner.learner.plot()
            else:
                yield plotter(runner.learner)

    streams = [hv.streams.Stream.define("Next")()]
    dm = hv.DynamicMap(plot_generator(), streams=streams)
    dm.cache_size = 1

    if normalize:
        # XXX: change when https://github.com/pyviz/holoviews/issues/3637
        # is fixed.
        dm = dm.map(lambda obj: obj.opts(framewise=True), hv.Element)

    cancel_button = ipywidgets.Button(
        description="cancel live-plot", layout=ipywidgets.Layout(width="150px")
    )

    # Could have used dm.periodic in the following, but this would either spin
    # off a thread (and learner is not threadsafe) or block the kernel.

    async def updater():
        event = lambda: hv.streams.Stream.trigger(  # noqa: E731
            dm.streams
        )  # XXX: used to be dm.event()
        # see https://github.com/pyviz/holoviews/issues/3564
        try:
            while not runner.task.done():
                event()
                await asyncio.sleep(update_interval)
            event()  # fire off one last update before we die
        finally:
            if active_plotting_tasks[name] is asyncio.current_task():
                active_plotting_tasks.pop(name, None)
            cancel_button.layout.display = "none"  # remove cancel button

    def cancel(_):
        with suppress(KeyError):
            active_plotting_tasks[name].cancel()

    active_plotting_tasks[name] = runner.ioloop.create_task(updater())
    cancel_button.on_click(cancel)

    display(cancel_button)
    return dm


def should_update(status):
    try:
        # Get the length of the write buffer size
        buffer_size = len(status.comm.kernel.iopub_thread._events)

        # Make sure to only keep all the messages when the notebook
        # is viewed, this means 'buffer_size == 1'. However, when not
        # viewing the notebook the buffer fills up. When this happens
        # we decide to only add messages to it when a certain probability.
        # i.e. we're offline for 12h, with an update_interval of 0.5s,
        # and without the reduced probability, we have buffer_size=86400.
        # With the correction this is np.log(86400) / np.log(1.1) = 119.2
        return 1.1**buffer_size * random.random() < 1
    except Exception:
        # We catch any Exception because we are using a private API.
        return True


def live_info(runner, *, update_interval=0.5):
    """Display live information about the runner.

    Returns an interactive ipywidget that can be
    visualized in a Jupyter notebook.
    """
    if not _holoviews_enabled:
        raise RuntimeError(
            "Live plotting is not enabled; did you run "
            "'adaptive.notebook_extension()'?"
        )

    import ipywidgets
    from IPython.display import display

    status = ipywidgets.HTML(value=_info_html(runner))

    cancel = ipywidgets.Button(
        description="cancel runner", layout=ipywidgets.Layout(width="100px")
    )
    cancel.on_click(lambda _: runner.cancel())

    async def update():
        while not runner.task.done():
            await asyncio.sleep(update_interval)

            if should_update(status):
                status.value = _info_html(runner)
            else:
                await asyncio.sleep(0.05)

        status.value = _info_html(runner)
        cancel.layout.display = "none"

    runner.ioloop.create_task(update())

    display(ipywidgets.VBox((status, cancel)))


def _table_row(i, key, value):
    """Style the rows of a table. Based on the default Jupyterlab table style."""
    style = "text-align: right; padding: 0.5em 0.5em; line-height: 1.0;"
    if i % 2 == 1:
        style += " background: var(--md-grey-100);"
    return f'<tr><th style="{style}">{key}</th><th style="{style}">{value}</th></tr>'


def _info_html(runner):
    status = runner.status()

    color = {
        "cancelled": "orange",
        "failed": "red",
        "running": "blue",
        "finished": "green",
    }[status]

    overhead = runner.overhead()
    red_level = max(0, min(int(255 * overhead / 100), 255))
    overhead_color = f"#{red_level:02x}{255 - red_level:02x}{0:02x}"

    info = [
        ("status", f'<font color="{color}">{status}</font>'),
        ("elapsed time", datetime.timedelta(seconds=runner.elapsed_time())),
        ("overhead", f'<font color="{overhead_color}">{overhead:.2f}%</font>'),
    ]

    with suppress(Exception):
        info.append(("# of points", runner.learner.npoints))

    with suppress(Exception):
        info.append(("# of samples", runner.learner.nsamples))

    with suppress(Exception):
        info.append(("latest loss", f'{runner.learner._cache["loss"]:.3f}'))

    table = "\n".join(_table_row(i, k, v) for i, (k, v) in enumerate(info))

    return f"""
        <table>
        {table}
        </table>
    """
