Tutorial `~adaptive.SequenceLearner`
---------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.SequenceLearner`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()

    import holoviews as hv
    import numpy as np

This learner will learn a sequence. It simply returns
the points in the provided sequence when asked.

This is useful when your problem cannot be formulated in terms of
another adaptive learner, but you still want to use Adaptive's
routines to run, (periodically) save, and plot.

.. jupyter-execute::

    from adaptive import SequenceLearner

    def f(x):
        return int(x) ** 2

    seq = np.linspace(-15, 15, 1000)
    learner = SequenceLearner(f, seq)

    runner = adaptive.Runner(learner, SequenceLearner.done)
    # that goal is same as `lambda learner: learner.done()`

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    def plotter(learner):
        data = learner.data if learner.data else []
        return hv.Scatter(data)

    runner.live_plot(plotter=plotter)

``learner.data`` contains a dictionary that maps the index of the point of ``learner.sequence`` to the value at that point.

To get the values in the same order as the input sequence (``learner.sequence``) use

.. jupyter-execute::

    result = learner.result()
    print(result[:10])  # print the 10 first values
