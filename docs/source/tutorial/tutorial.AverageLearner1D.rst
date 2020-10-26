Tutorial `~adaptive.AverageLearner1D`
------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.AverageLearner1D`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()
    %config InlineBackend.figure_formats=set(['svg'])

    import numpy as np
    from functools import partial
    import random

General use
..........................

First, we define the (noisy) function to be sampled. Note that the parameter
``sigma`` corresponds to the standard deviation of the Gaussian noise.

.. jupyter-execute::

    def f(x, sigma=0, peak_width=0.05, offset=-0.5, wait=False):
        from time import sleep
        from random import random

        if wait:
            sleep(random())

        function = x ** 3 - x + 3 * peak_width ** 2 / (peak_width ** 2 + (x - offset) ** 2)
        return function + np.random.normal(0, sigma)

This is how the function looks in the absence of noise:

.. jupyter-execute::

    import matplotlib.pyplot as plt
    x = np.linspace(-2,2,500)
    plt.plot(x, f(x, sigma=0));

This is how a single realization of the noisy function looks:

.. jupyter-execute::

    plt.plot(x, [f(xi, sigma=1) for xi in x]);

To obtain an estimate of the mean value of the function at each point ``x``, we
take many samples at ``x`` and calculate the sample mean. The learner will
autonomously determine whether the next samples should be taken at an old
point (to improve the estimate of the mean at that point) or at a new one.

We start by initializing a 1D average learner:

.. jupyter-execute::

    learner = adaptive.AverageLearner1D(
        function=partial(f, sigma=1),
        bounds=(-2,2))

As with other types of learners, we need to initialize a runner with a certain
goal to run our learner. In this case, we set 10000 samples as the goal (the
second condition ensures that we have at least 20 samples at each point):

.. jupyter-execute::

    runner = adaptive.Runner(learner, goal=lambda l: l.total_samples >= 10000 and min(l._number_samples.values()) >= 20)
    runner.live_info()
    runner.live_plot(update_interval=0.1)

Fine tuning
..........................

In some cases, the default configuration of the 1D average learner can be
sub-optimal. One can then tune the internal parameters of the learner. The most
relevant are:

- ``loss_per_interval``: loss function (see Learner1D).
- ``delta``: this parameter is the most relevant and controls the balance between resampling existing points (exploitation) and sampling new ones (exploration). Its value should remain between 0 and 1 (the default value is 0.2). Large values favor the "exploration" behavior, although this can make the learner to sample noise. Small values favor the "exploitation" behavior, leading the learner to thoroughly resample existing points. In general, the optimal value of ``delta`` is between 0.1 and 0.4.
- ``neighbor_sampling``: each new point is initially sampled a fraction ``neighbor_sampling`` of the number of samples of its nearest neighbor. We recommend to keep the value of ``neighbor_sampling`` below 1 to prevent oversampling.
- ``min_samples``: minimum number of samples that are initially taken at a new point. This parameter can prevent the learner from sampling noise in case we accidentally set a too large value of ``delta``.
- ``max_samples``: maximum number of samples at each point. If a point has been sampled ``max_samples`` times, it will not be sampled again. This prevents the "exploitation" to drastically dominate over the "exploration" behavior in case we set a too small ``delta``.
- ``min_error``: minimum uncertainty at each point (this uncertainty corresponds to the standard deviation in the estimate of the mean). As ``max_samples``, this parameter can prevent the "exploitation" to drastically dominate over the "exploration" behavior.

As an example, assume that we wanted to resample the points from the previous
learner. We can decrease ``delta`` to 0.1 and set ``min_error`` to 0.05 if we do
not require accuracy beyond this value:

.. jupyter-execute::

    learner.delta = 0.1
    learner.min_error = 0.05

    runner = adaptive.Runner(learner, goal=lambda l: l.total_samples >= 20000 and min(l._number_samples.values()) >= 20)
    runner.live_info()
    runner.live_plot(update_interval=0.1)

On the contrary, if we want to push forward the "exploration", we can set a larger
``delta`` and limit the maximum number of samples taken at each point:

.. jupyter-execute::

    learner.delta = 0.3
    learner.max_samples = 1000

    runner = adaptive.Runner(learner, goal=lambda l: l.total_samples >= 25000 and min(l._number_samples.values()) >= 20)
    runner.live_info()
    runner.live_plot(update_interval=0.1)
