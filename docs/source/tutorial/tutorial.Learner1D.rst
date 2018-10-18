Tutorial `~adaptive.Learner1D`
------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`Learner1D`

.. execute::
    :hide-code:
    :new-notebook: Learner1D

    import adaptive
    adaptive.notebook_extension()

    import numpy as np
    from functools import partial
    import random

scalar output: ``f:ℝ → ℝ``
..........................

We start with the most common use-case: sampling a 1D function
:math:`\ f: ℝ → ℝ`.

We will use the following function, which is a smooth (linear)
background with a sharp peak at a random location:

.. execute::

    offset = random.uniform(-0.5, 0.5)

    def f(x, offset=offset, wait=True):
        from time import sleep
        from random import random

        a = 0.01
        if wait:
            sleep(random())
        return x + a**2 / (a**2 + (x - offset)**2)

We start by initializing a 1D “learner”, which will suggest points to
evaluate, and adapt its suggestions as more and more points are
evaluated.

.. execute::

    learner = adaptive.Learner1D(f, bounds=(-1, 1))

Next we create a “runner” that will request points from the learner and
evaluate ‘f’ on them.

By default on Unix-like systems the runner will evaluate the points in
parallel using local processes `concurrent.futures.ProcessPoolExecutor`.

On Windows systems the runner will try to use a `distributed.Client`
if `distributed` is installed. A `~concurrent.futures.ProcessPoolExecutor`
cannot be used on Windows for reasons.

.. execute::

    # The end condition is when the "loss" is less than 0.1. In the context of the
    # 1D learner this means that we will resolve features in 'func' with width 0.1 or wider.
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)

.. execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

When instantiated in a Jupyter notebook the runner does its job in the
background and does not block the IPython kernel. We can use this to
create a plot that updates as new data arrives:

.. execute::

    runner.live_info()

.. execute::

    runner.live_plot(update_interval=0.1)

We can now compare the adaptive sampling to a homogeneous sampling with
the same number of points:

.. execute::

    if not runner.task.done():
        raise RuntimeError('Wait for the runner to finish before executing the cells below!')

.. execute::

    learner2 = adaptive.Learner1D(f, bounds=learner.bounds)

    xs = np.linspace(*learner.bounds, len(learner.data))
    learner2.tell_many(xs, map(partial(f, wait=False), xs))

    learner.plot() + learner2.plot()


vector output: ``f:ℝ → ℝ^N``
............................

Sometimes you may want to learn a function with vector output:

.. execute::

    random.seed(0)
    offsets = [random.uniform(-0.8, 0.8) for _ in range(3)]

    # sharp peaks at random locations in the domain
    def f_levels(x, offsets=offsets):
        a = 0.01
        return np.array([offset + x + a**2 / (a**2 + (x - offset)**2)
                         for offset in offsets])

``adaptive`` has you covered! The ``Learner1D`` can be used for such
functions:

.. execute::

    learner = adaptive.Learner1D(f_levels, bounds=(-1, 1))
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)

.. execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. execute::

    runner.live_info()

.. execute::

    runner.live_plot(update_interval=0.1)
