Tutorial `~adaptive.SKOptLearner`
---------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.SKOptLearner`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()

    import holoviews as hv
    import numpy as np

We have wrapped the ``Optimizer`` class from
`scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`__,
to show how existing libraries can be integrated with ``adaptive``.

The ``SKOptLearner`` attempts to “optimize” the given function ``g``
(i.e. find the global minimum of ``g`` in the window of interest).

Here we use the same example as in the ``scikit-optimize``
`tutorial <https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/ask-and-tell.ipynb>`__.
Although ``SKOptLearner`` can optimize functions of arbitrary
dimensionality, we can only plot the learner if a 1D function is being
learned.

.. jupyter-execute::

    def F(x, noise_level=0.1):
        return (np.sin(5 * x) * (1 - np.tanh(x ** 2))
                + np.random.randn() * noise_level)

.. jupyter-execute::

    learner = adaptive.SKOptLearner(F, dimensions=[(-2., 2.)],
                                    base_estimator="GP",
                                    acq_func="gp_hedge",
                                    acq_optimizer="lbfgs",
                                   )
    runner = adaptive.Runner(learner, ntasks=1, goal=lambda l: l.npoints > 40)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    %%opts Overlay [legend_position='top']
    xs = np.linspace(*learner.space.bounds[0])
    to_learn = hv.Curve((xs, [F(x, 0) for x in xs]), label='to learn')

    runner.live_plot().relabel('prediction', depth=2) * to_learn
