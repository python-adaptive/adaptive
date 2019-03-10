Tutorial `~adaptive.Learner1D`
------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.Learner1D`

.. jupyter-execute::
    :hide-code:

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

.. jupyter-execute::

    offset = random.uniform(-0.5, 0.5)

    def f(x, offset=offset, wait=True):
        from time import sleep
        from random import random

        a = 0.01
        if wait:
            sleep(random() / 10)
        return x + a**2 / (a**2 + (x - offset)**2)

We start by initializing a 1D “learner”, which will suggest points to
evaluate, and adapt its suggestions as more and more points are
evaluated.

.. jupyter-execute::

    learner = adaptive.Learner1D(f, bounds=(-1, 1))

Next we create a “runner” that will request points from the learner and
evaluate ‘f’ on them.

By default on Unix-like systems the runner will evaluate the points in
parallel using local processes `concurrent.futures.ProcessPoolExecutor`.

On Windows systems the runner will try to use a `distributed.Client`
if `distributed` is installed. A `~concurrent.futures.ProcessPoolExecutor`
cannot be used on Windows for reasons.

.. jupyter-execute::

    # The end condition is when the "loss" is less than 0.1. In the context of the
    # 1D learner this means that we will resolve features in 'func' with width 0.1 or wider.
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

When instantiated in a Jupyter notebook the runner does its job in the
background and does not block the IPython kernel. We can use this to
create a plot that updates as new data arrives:

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    runner.live_plot(update_interval=0.1)

We can now compare the adaptive sampling to a homogeneous sampling with
the same number of points:

.. jupyter-execute::

    if not runner.task.done():
        raise RuntimeError('Wait for the runner to finish before executing the cells below!')

.. jupyter-execute::

    learner2 = adaptive.Learner1D(f, bounds=learner.bounds)

    xs = np.linspace(*learner.bounds, len(learner.data))
    learner2.tell_many(xs, map(partial(f, wait=False), xs))

    learner.plot() + learner2.plot()


vector output: ``f:ℝ → ℝ^N``
............................

Sometimes you may want to learn a function with vector output:

.. jupyter-execute::

    random.seed(0)
    offsets = [random.uniform(-0.8, 0.8) for _ in range(3)]

    # sharp peaks at random locations in the domain
    def f_levels(x, offsets=offsets):
        a = 0.01
        return np.array([offset + x + a**2 / (a**2 + (x - offset)**2)
                         for offset in offsets])

``adaptive`` has you covered! The ``Learner1D`` can be used for such
functions:

.. jupyter-execute::

    learner = adaptive.Learner1D(f_levels, bounds=(-1, 1))
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    runner.live_plot(update_interval=0.1)


Looking at curvature
....................

By default ``adaptive`` will sample more points where the (normalized)
euclidean distance between the neighboring points is large.
You may achieve better results sampling more points in regions with high
curvature. To do this, you need to tell the learner to look at the curvature
by specifying ``loss_per_interval``.

.. jupyter-execute::

    from adaptive.learner.learner1D import (curvature_loss_function,
                                            uniform_loss,
                                            default_loss)
    curvature_loss = curvature_loss_function()
    learner = adaptive.Learner1D(f, bounds=(-1, 1), loss_per_interval=curvature_loss)
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    runner.live_plot(update_interval=0.1)

We may see the difference of homogeneous sampling vs only one interval vs
including nearest neighboring intervals in this plot: We will look at 100 points.

.. jupyter-execute::

    def sin_exp(x):
        from math import exp, sin
        return sin(15 * x) * exp(-x**2*2)

    learner_h = adaptive.Learner1D(sin_exp, (-1, 1), loss_per_interval=uniform_loss)
    learner_1 = adaptive.Learner1D(sin_exp, (-1, 1), loss_per_interval=default_loss)
    learner_2 = adaptive.Learner1D(sin_exp, (-1, 1), loss_per_interval=curvature_loss)

    npoints_goal = lambda l: l.npoints >= 100
    # adaptive.runner.simple is a non parallel blocking runner.
    adaptive.runner.simple(learner_h, goal=npoints_goal)
    adaptive.runner.simple(learner_1, goal=npoints_goal)
    adaptive.runner.simple(learner_2, goal=npoints_goal)

    (learner_h.plot().relabel('homogeneous')
     + learner_1.plot().relabel('euclidean loss')
     + learner_2.plot().relabel('curvature loss')).cols(2)

More info about using custom loss functions can be found
in :ref:`Custom adaptive logic for 1D and 2D`.
