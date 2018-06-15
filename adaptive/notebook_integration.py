# -*- coding: utf-8 -*-
import importlib
import asyncio
import datetime
from pkg_resources import parse_version
import warnings


_async_enabled = False
_plotting_enabled = False


def notebook_extension():
    if not in_ipynb():
        raise RuntimeError('"adaptive.notebook_extension()" may only be run '
                           'from a Jupyter notebook.')

    global _plotting_enabled
    _plotting_enabled = False
    try:
        import ipywidgets
        import holoviews
        holoviews.notebook_extension('bokeh')
        _plotting_enabled = True
    except ModuleNotFoundError:
        warnings.warn("holoviews and (or) ipywidgets are not installed; plotting "
                      "is disabled.", RuntimeWarning)

    global _async_enabled
    get_ipython().magic('gui asyncio')
    _async_enabled = True


def ensure_holoviews():
    try:
        return importlib.import_module('holoviews')
    except ModuleNotFounderror:
        raise RuntimeError('holoviews is not installed; plotting is disabled.')


def in_ipynb():
    try:
        # If we are running in IPython, then `get_ipython()` is always a global
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        return False


# Fancy displays in the Jupyter notebook

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
    if not _plotting_enabled:
        raise RuntimeError("Live plotting is not enabled; did you run "
                           "'adaptive.notebook_extension()'?")

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

    dm = hv.DynamicMap(plot_generator(),
                       streams=[hv.streams.Stream.define('Next')()])
    cancel_button = ipywidgets.Button(description='cancel live-plot',
                                      layout=ipywidgets.Layout(width='150px'))

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
            cancel_button.layout.display = 'none'  # remove cancel button

    def cancel(_):
        try:
            active_plotting_tasks[name].cancel()
        except KeyError:
            pass

    active_plotting_tasks[name] = runner.ioloop.create_task(updater())
    cancel_button.on_click(cancel)

    display(cancel_button)
    return dm


def live_info(runner, *, update_interval=0.5):
    """Display live information about the runner.

    Returns an interactive ipywidget that can be
    visualized in a Jupyter notebook.
    """
    if not _plotting_enabled:
        raise RuntimeError("Live plotting is not enabled; did you run "
                           "'adaptive.notebook_extension()'?")

    import ipywidgets
    from IPython.display import display

    status = ipywidgets.HTML(value=_info_html(runner))

    cancel = ipywidgets.Button(description='cancel runner',
                               layout=ipywidgets.Layout(width='100px'))
    cancel.on_click(lambda _: runner.cancel())

    async def update():
        while not runner.task.done():
            await asyncio.sleep(update_interval)
            status.value = _info_html(runner)
        status.value = _info_html(runner)
        cancel.layout.display = 'none'

    runner.ioloop.create_task(update())

    display(ipywidgets.HBox(
        (status, cancel),
        layout=ipywidgets.Layout(border='solid 1px',
                                 width='200px',
                                 align_items='center'),
    ))


def _info_html(runner):
    status = runner.status()

    color = {'cancelled': 'orange',
             'failed': 'red',
             'running': 'blue',
             'finished': 'green'}[status]

    t_total = runner.elapsed_time()
    efficiency = (t_total - runner.time_ask_tell) / t_total * 100

    info = [
        ('status', f'<font color="{color}">{status}</font>'),
        ('elapsed time', datetime.timedelta(seconds=t_total)),
        (f'efficiency', f'{efficiency:.1f}%'),
    ]

    try:
        info.append(('# of points', runner.learner.npoints))
    except Exception:
        pass

    template = '<dt>{}</dt><dd>{}</dd>'
    table = '\n'.join(template.format(k, v) for k, v in info)

    return f'''
        <dl>
        {table}
        </dl>
    '''
