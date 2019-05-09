Tutorial AverageLearners (0D, 1D, and 2D)
-----------------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.AverageLearners`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension(_inline_js=False)

`~adaptive.AverageLearner` (0D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next type of learner averages a function until the uncertainty in
the average meets some condition.

This is useful for sampling a random variable. The function passed to
the learner must formally take a single parameter, which should be used
like a “seed” for the (pseudo-) random variable (although in the current
implementation the seed parameter can be ignored by the function).

.. jupyter-execute::

    def g(n):
        import random
        from time import sleep
        sleep(random.random() / 1000)
        # Properly save and restore the RNG state
        state = random.getstate()
        random.seed(n)
        val = random.gauss(0.5, 1)
        random.setstate(state)
        return val

.. jupyter-execute::

    learner = adaptive.AverageLearner(g, atol=None, rtol=0.05)
    # `loss < 1` means that we reached the `rtol` or `atol`
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 1)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    runner.live_plot(update_interval=0.1)

`~adaptive.AverageLearner1D` and `~adaptive.AverageLearner2D`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This learner is a combination between the `~adaptive.Learner1D` (or `~adaptive.Learner2D`)
and the `~adaptive.AverageLearner`, in a way such that it handles
stochastic functions with one (or two) variables.

Here, when chosing points the learner can either:

* add more values/seeds to existing points
* add more intervals (or triangles)

So, the ``learner`` compares **the loss of intervals (or triangles)** with the **standard error** of an existing point.

The relative importance of both can be adjusted by a hyperparameter ``learner.average_priority``, see the doc-string for more information.

See the following plot for a visual explanation.

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    %matplotlib inline
    rcParams['figure.dpi'] = 300
    rcParams['text.usetex'] = True

    np.random.seed(1)
    xs = np.sort(np.random.uniform(-1, 1, 3))
    errs = np.abs(np.random.randn(3))
    ys = xs**3
    means = lambda x: np.convolve(x, np.ones(2) / 2, mode='valid')
    xs_means = means(xs)
    ys_means = means(ys)

    fig, ax = plt.subplots()
    plt.scatter(xs, ys, c='k')
    ax.errorbar(xs, ys, errs, capsize=5, c='k')
    ax.annotate(
        s=r'$L_{1,2} = \sqrt{\Delta x^2 + \Delta \bar{y}^2}$',
        xy=(np.mean([xs[0], xs[1], xs[1]]),
            np.mean([ys[0], ys[1], ys[1]])),
        xytext=(xs_means[0], ys_means[0] + 1),
        arrowprops=dict(arrowstyle='->'),
        ha='center',
    )

    for i, (x, y, err) in enumerate(zip(xs, ys, errs)):
        err_str = fr'${{\sigma}}_{{\bar {{y}}_{i+1}}}$'
        ax.annotate(
            s=err_str,
            xy=(x, y + err/2),
            xytext=(x + 0.1, y + err + 0.5),
            arrowprops=dict(arrowstyle='->'),
            ha='center',
        )

        ax.annotate(
            s=fr'$x_{i+1}, \bar{{y}}_{i+1}$',
            xy=(x, y),
            xytext=(x + 0.1, y - 0.5),
            arrowprops=dict(arrowstyle='->'),
            ha='center',
        )


    ax.scatter(xs, ys, c='green', s=5, zorder=5, label='more seeds')
    ax.scatter(xs_means, ys_means, c='red', s=5, zorder=5, label='new point')
    ax.legend()

    ax.text(
        x=0.5,
        y=0.0,
        s=(r'$\textrm{if}\; \max{(L_{i,i+1})} > \textrm{average\_priority} \cdot \max{\sigma_{\bar{y}_{i}}} \rightarrow,\;\textrm{add new point}$'
           '\n'
           r'$\textrm{if}\; \max{(L_{i,i+1})} < \textrm{average\_priority} \cdot \max{\sigma_{\bar{y}_{i}}} \rightarrow,\;\textrm{add new seeds}$'),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes
    )
    ax.set_title('AverageLearner1D')
    ax.axis('off')
    plt.show()


In this plot :math:`L_{i,i+1}` is the default ``learner.loss_per_interval`` and :math:`\sigma_{\bar{y}_i}` is the standard error of the mean.

Basically, we put all losses per interval and standard errors (scaled by ``average_priority``) in a list.
The point of the maximal value will be chosen.

It is important to note that all :math:`x`, :math:`y`, (and :math:`z` in 2D) are scaled to be inside
the unit square (or cube) in both the ``loss_per_interval`` and the standard error.


.. warning::
    If you choose the ``average_priority`` too low, the standard errors :math:`\sigma_{\bar{y}_i}` will be high.
    This leads to incorrectly estimated averages :math:`\bar{y}_i` and therefore points that are closeby, can appear to be far away.
    This in turn results in new points unnecessarily being added and an unstable sampling algorithm!


Let's again try to learn some functions but now with uniform (and `heteroscedastic <https://en.wikipedia.org/wiki/Heteroscedasticity>`_ in 2D) noise. We start with 1D and then go to 2D.

`~adaptive.AverageLearner1D`
............................

.. jupyter-execute::

    def noisy_peak(x_seed):
        import random
        x, seed = x_seed
        random.seed(x_seed)  # to make the random function deterministic
        a = 0.01
        peak = x + a**2 / (a**2 + x**2)
        noise = random.uniform(-0.5, 0.5)
        return peak + noise

    learner = adaptive.AverageLearner1D(noisy_peak, bounds=(-1, 1), average_priority=40)
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.05)
    runner.live_info()

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    %%opts Image {+axiswise} [colorbar=True]
    # We plot the average

    def plotter(learner):
        plot = learner.plot()
        number_of_points = learner.mean_values_per_point()
        title = f'loss={learner.loss():.3f}, mean_npoints={number_of_points}'
        return plot.opts(plot=dict(title_format=title))

    runner.live_plot(update_interval=0.1, plotter=plotter)

`~adaptive.AverageLearner2D`
............................

.. jupyter-execute::

    def noisy_ring(xy_seed):
        import numpy as np
        import random
        (x, y), seed = xy_seed
        random.seed(xy_seed)  # to make the random function deterministic
        a = 0.2
        z = (x**2 + y**2 - 0.75**2) / a**2
        plateau = np.arctan(z)
        noise = random.uniform(-2, 2) * np.exp(-z**2)
        return plateau + noise

    learner = adaptive.AverageLearner2D(noisy_ring, bounds=[(-1, 1), (-1, 1)])
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)
    runner.live_info()

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

See the average number of values per point with:

.. jupyter-execute::

    learner.mean_values_per_point()

Let's plot the average and the number of values per point.
Because the noise lies on a circle we expect the number of values per
to be higher on the circle.

.. jupyter-execute::

    %%opts Image {+axiswise} [colorbar=True]
    # We plot the average and the standard deviation
    def plotter(learner):
        return (learner.plot_std_or_n('mean')
                + learner.plot_std_or_n('std')
                + learner.plot_std_or_n('n')).cols(2)

    runner.live_plot(update_interval=0.1, plotter=plotter)
