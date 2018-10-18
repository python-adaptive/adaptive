Advanced Topics
===============

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`advanced-topics`

.. execute::
    :hide-code:
    :new-notebook: advanced-topics

    import adaptive
    adaptive.notebook_extension()

    from functools import partial
    import random

    offset = random.uniform(-0.5, 0.5)

    def f(x, offset=offset, wait=True):
        from time import sleep
        from random import random

        a = 0.01
        if wait:
            sleep(random())
        return x + a**2 / (a**2 + (x - offset)**2)


A watched pot never boils!
--------------------------

`adaptive.Runner` does its work in an `asyncio` task that runs
concurrently with the IPython kernel, when using ``adaptive`` from a
Jupyter notebook. This is advantageous because it allows us to do things
like live-updating plots, however it can trip you up if you’re not
careful.

Notably: **if you block the IPython kernel, the runner will not do any
work**.

For example if you wanted to wait for a runner to complete, **do not
wait in a busy loop**:

.. code:: python

   while not runner.task.done():
       pass

If you do this then **the runner will never finish**.

What to do if you don’t care about live plotting, and just want to run
something until its done?

The simplest way to accomplish this is to use
`adaptive.BlockingRunner`:

.. execute::

    learner = adaptive.Learner1D(partial(f, wait=False), bounds=(-1, 1))
    adaptive.BlockingRunner(learner, goal=lambda l: l.loss() < 0.01)
    # This will only get run after the runner has finished
    learner.plot()

Reproducibility
---------------

By default ``adaptive`` runners evaluate the learned function in
parallel across several cores. The runners are also opportunistic, in
that as soon as a result is available they will feed it to the learner
and request another point to replace the one that just finished.

Because the order in which computations complete is non-deterministic,
this means that the runner behaves in a non-deterministic way. Adaptive
makes this choice because in many cases the speedup from parallel
execution is worth sacrificing the “purity” of exactly reproducible
computations.

Nevertheless it is still possible to run a learner in a deterministic
way with adaptive.

The simplest way is to use `adaptive.runner.simple` to run your
learner:

.. execute::

    learner = adaptive.Learner1D(partial(f, wait=False), bounds=(-1, 1))

    # blocks until completion
    adaptive.runner.simple(learner, goal=lambda l: l.loss() < 0.01)

    learner.plot()

Note that unlike `adaptive.Runner`, `adaptive.runner.simple`
*blocks* until it is finished.

If you want to enable determinism, want to continue using the
non-blocking `adaptive.Runner`, you can use the
`adaptive.runner.SequentialExecutor`:

.. execute::

    from adaptive.runner import SequentialExecutor

    learner = adaptive.Learner1D(partial(f, wait=False), bounds=(-1, 1))

    runner = adaptive.Runner(learner, executor=SequentialExecutor(), goal=lambda l: l.loss() < 0.01)

.. execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. execute::

    runner.live_info()

.. execute::

    runner.live_plot(update_interval=0.1)

Cancelling a runner
-------------------

Sometimes you want to interactively explore a parameter space, and want
the function to be evaluated at finer and finer resolution and manually
control when the calculation stops.

If no ``goal`` is provided to a runner then the runner will run until
cancelled.

``runner.live_info()`` will provide a button that can be clicked to stop
the runner. You can also stop the runner programatically using
``runner.cancel()``.

.. execute::

    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    runner = adaptive.Runner(learner)

.. execute::
    :hide-code:

    import asyncio
    await asyncio.sleep(3)  # This is not needed in the notebook!

.. execute::

    runner.cancel()  # Let's execute this after 3 seconds

.. execute::

    runner.live_info()

.. execute::

    runner.live_plot(update_interval=0.1)

.. execute::

    print(runner.status())

Debugging Problems
------------------

Runners work in the background with respect to the IPython kernel, which
makes it convenient, but also means that inspecting errors is more
difficult because exceptions will not be raised directly in the
notebook. Often the only indication you will have that something has
gone wrong is that nothing will be happening.

Let’s look at the following example, where the function to be learned
will raise an exception 10% of the time.

.. execute::

    def will_raise(x):
        from random import random
        from time import sleep

        sleep(random())
        if random() < 0.1:
            raise RuntimeError('something went wrong!')
        return x**2

    learner = adaptive.Learner1D(will_raise, (-1, 1))
    runner = adaptive.Runner(learner)  # without 'goal' the runner will run forever unless cancelled


.. execute::
    :hide-code:

    import asyncio
    await asyncio.sleep(4)  # in 4 seconds it will surely have failed

.. execute::

    runner.live_info()

.. execute::

    runner.live_plot()

The above runner should continue forever, but we notice that it stops
after a few points are evaluated.

First we should check that the runner has really finished:

.. execute::

    runner.task.done()

If it has indeed finished then we should check the ``result`` of the
runner. This should be ``None`` if the runner stopped successfully. If
the runner stopped due to an exception then asking for the result will
raise the exception with the stack trace:

.. execute::

    runner.task.result()

Logging runners
~~~~~~~~~~~~~~~

Runners do their job in the background, which makes introspection quite
cumbersome. One way to inspect runners is to instantiate one with
``log=True``:

.. execute::

    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.1,
                             log=True)

.. execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. execute::

    runner.live_info()

This gives a the runner a ``log`` attribute, which is a list of the
``learner`` methods that were called, as well as their arguments. This
is useful because executors typically execute their tasks in a
non-deterministic order.

This can be used with `adaptive.runner.replay_log` to perfom the same
set of operations on another runner:

.. execute::

    reconstructed_learner = adaptive.Learner1D(f, bounds=learner.bounds)
    adaptive.runner.replay_log(reconstructed_learner, runner.log)

.. execute::

    learner.plot().Scatter.I.opts(style=dict(size=6)) * reconstructed_learner.plot()

Timing functions
~~~~~~~~~~~~~~~~

To time the runner you **cannot** simply use

.. code:: python

   now = datetime.now()
   runner = adaptive.Runner(...)
   print(datetime.now() - now)

because this will be done immediately. Also blocking the kernel with
``while not runner.task.done()`` will not work because the runner will
not do anything when the kernel is blocked.

Therefore you need to create an ``async`` function and hook it into the
``ioloop`` like so:

.. execute::

    import asyncio

    async def time(runner):
        from datetime import datetime
        now = datetime.now()
        await runner.task
        return datetime.now() - now

    ioloop = asyncio.get_event_loop()

    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)

    timer = ioloop.create_task(time(runner))

.. execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. execute::

    # The result will only be set when the runner is done.
    timer.result()

Using Runners from a script
---------------------------

Runners can also be used from a Python script independently of the
notebook.

The simplest way to accomplish this is simply to use the
`~adaptive.BlockingRunner`:

.. code:: python

   import adaptive

   def f(x):
       return x

   learner = adaptive.Learner1D(f, (-1, 1))

   adaptive.BlockingRunner(learner, goal=lambda: l: l.loss() < 0.1)

If you use `asyncio` already in your script and want to integrate
``adaptive`` into it, then you can use the default `~adaptive.Runner` as you
would from a notebook. If you want to wait for the runner to finish,
then you can simply

.. code:: python

       await runner.task

from within a coroutine.
