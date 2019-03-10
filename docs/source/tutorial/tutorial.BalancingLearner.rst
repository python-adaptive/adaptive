Tutorial `~adaptive.BalancingLearner`
-------------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.BalancingLearner`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()

    import holoviews as hv
    import numpy as np
    from functools import partial
    import random

The balancing learner is a “meta-learner” that takes a list of learners.
When you request a point from the balancing learner, it will query all
of its “children” to figure out which one will give the most
improvement.

The balancing learner can for example be used to implement a poor-man’s
2D learner by using the `~adaptive.Learner1D`.

.. jupyter-execute::

    def h(x, offset=0):
        a = 0.01
        return x + a**2 / (a**2 + (x - offset)**2)

    learners = [adaptive.Learner1D(partial(h, offset=random.uniform(-1, 1)),
                bounds=(-1, 1)) for i in range(10)]

    bal_learner = adaptive.BalancingLearner(learners)
    runner = adaptive.Runner(bal_learner, goal=lambda l: l.loss() < 0.01)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    plotter = lambda learner: hv.Overlay([L.plot() for L in learner.learners])
    runner.live_plot(plotter=plotter, update_interval=0.1)

Often one wants to create a set of ``learner``\ s for a cartesian
product of parameters. For that particular case we’ve added a
``classmethod`` called `~adaptive.BalancingLearner.from_product`.
See how it works below

.. jupyter-execute::

    from scipy.special import eval_jacobi

    def jacobi(x, n, alpha, beta): return eval_jacobi(n, alpha, beta, x)

    combos = {
        'n': [1, 2, 4, 8],
        'alpha': np.linspace(0, 2, 3),
        'beta': np.linspace(0, 1, 5),
    }

    learner = adaptive.BalancingLearner.from_product(
        jacobi, adaptive.Learner1D, dict(bounds=(0, 1)), combos)

    runner = adaptive.BlockingRunner(learner, goal=lambda l: l.loss() < 0.01)

    # The `cdims` will automatically be set when using `from_product`, so
    # `plot()` will return a HoloMap with correctly labeled sliders.
    learner.plot().overlay('beta').grid().select(y=(-1, 3))
