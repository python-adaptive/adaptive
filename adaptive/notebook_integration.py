# -*- coding: utf-8 -*-
import asyncio

from IPython import get_ipython
from ipykernel.eventloops import register_integration
import holoviews as hv


# IPython event loop integraion

@register_integration('asyncio')
def _loop_asyncio(kernel):
    '''Start a kernel with asyncio event loop support.'''
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
    return hv.notebook_extension('bokeh')


# Plotting

def live_plot(runner, *, plotter=None, update_interval=2):

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
        while not runner.task.done():
            dm.event()
            await asyncio.sleep(update_interval)
        dm.event()  # fire off one last update before we die

    # Fire and forget -- the task will die anyway once the runner has finished.
    asyncio.get_event_loop().create_task(updater())

    return dm
