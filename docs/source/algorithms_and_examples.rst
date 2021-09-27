.. include:: ../../README.rst
    :start-after: summary-end
    :end-before: not-in-documentation-start

- `~adaptive.Learner1D`, for 1D functions ``f: ℝ → ℝ^N``,
- `~adaptive.Learner2D`, for 2D functions ``f: ℝ^2 → ℝ^N``,
- `~adaptive.LearnerND`, for ND functions ``f: ℝ^N → ℝ^M``,
- `~adaptive.AverageLearner`, for random variables where you want to
  average the result over many evaluations,
- `~adaptive.AverageLearner1D`, for stochastic 1D functions where you want to
  estimate the mean value of the function at each point,
- `~adaptive.IntegratorLearner`, for
  when you want to intergrate a 1D function ``f: ℝ → ℝ``.
- `~adaptive.BalancingLearner`, for when you want to run several learners at once,
  selecting the “best” one each time you get more points.

Meta-learners (to be used with other learners):

- `~adaptive.BalancingLearner`, for when you want to run several learners at once,
  selecting the “best” one each time you get more points,
- `~adaptive.DataSaver`, for when your function doesn't just return a scalar or a vector.

In addition to the learners, ``adaptive`` also provides primitives for
running the sampling across several cores and even several machines,
with built-in support for
`concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_,
`mpi4py <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html>`_,
`loky <https://loky.readthedocs.io/en/stable/>`_,
`ipyparallel <https://ipyparallel.readthedocs.io/en/latest/>`_ and
`distributed <https://distributed.readthedocs.io/en/latest/>`_.

Examples
--------

Here are some examples of how Adaptive samples vs. homogeneous sampling. Click
on the *Play* :fa:`play` button or move the sliders.

.. jupyter-execute::
    :hide-code:

    import itertools
    import adaptive
    from adaptive.learner.learner1D import uniform_loss, default_loss
    import holoviews as hv
    import numpy as np

    adaptive.notebook_extension()
    hv.output(holomap="scrubber")

`adaptive.Learner1D`
~~~~~~~~~~~~~~~~~~~~

Adaptively learning a 1D function (the plot below) and live-plotting the process in a Jupyter notebook is as easy as

.. code:: python

    from adaptive import notebook_extension, Runner, Learner1D
    notebook_extension()  # enables notebook integration

    def peak(x, a=0.01):  # function to "learn"
        return x + a**2 / (a**2 + x**2)

    learner = Learner1D(peak, bounds=(-1, 1))

    def goal(learner):
        return learner.loss() < 0.01  # continue until loss is small enough

    runner = Runner(learner, goal)  # start calculation on all CPU cores
    runner.live_info()  # shows a widget with status information
    runner.live_plot()


.. jupyter-execute::
    :hide-code:

    def f(x, offset=0.07357338543088588):
        a = 0.01
        return x + a**2 / (a**2 + (x - offset)**2)

    def plot_loss_interval(learner):
        if learner.npoints >= 2:
            x_0, x_1 = max(learner.losses, key=learner.losses.get)
            y_0, y_1 = learner.data[x_0], learner.data[x_1]
            x, y = [x_0, x_1], [y_0, y_1]
        else:
            x, y = [], []
        return hv.Scatter((x, y)).opts(style=dict(size=6, color="r"))

    def plot(learner, npoints):
        adaptive.runner.simple(learner, lambda l: l.npoints == npoints)
        return (learner.plot() * plot_loss_interval(learner))[:, -1.1:1.1]

    def get_hm(loss_per_interval, N=101):
        learner = adaptive.Learner1D(f, bounds=(-1, 1), loss_per_interval=loss_per_interval)
        plots = {n: plot(learner, n) for n in range(N)}
        return hv.HoloMap(plots, kdims=["npoints"])

    layout = (
        get_hm(uniform_loss).relabel("homogeneous samping")
        + get_hm(default_loss).relabel("with adaptive")
    )

    layout.opts(plot=dict(toolbar=None))

`adaptive.Learner2D`
~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    def ring(xy):
        import numpy as np
        x, y = xy
        a = 0.2
        return x + np.exp(-(x**2 + y**2 - 0.75**2)**2/a**4)

    def plot(learner, npoints):
        adaptive.runner.simple(learner, lambda l: l.npoints == npoints)
        learner2 = adaptive.Learner2D(ring, bounds=learner.bounds)
        xs = ys = np.linspace(*learner.bounds[0], int(learner.npoints**0.5))
        xys = list(itertools.product(xs, ys))
        learner2.tell_many(xys, map(ring, xys))
        return (learner2.plot().relabel('homogeneous grid')
                + learner.plot().relabel('with adaptive')
                + learner2.plot(tri_alpha=0.5).relabel('homogeneous sampling')
                + learner.plot(tri_alpha=0.5).relabel('with adaptive')).cols(2)

    learner = adaptive.Learner2D(ring, bounds=[(-1, 1), (-1, 1)])
    plots = {n: plot(learner, n) for n in range(4, 1010, 20)}
    hv.HoloMap(plots, kdims=['npoints']).collate()

`adaptive.AverageLearner`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    def g(n):
        import random
        random.seed(n)
        val = random.gauss(0.5, 0.5)
        return val

    learner = adaptive.AverageLearner(g, atol=None, rtol=0.01)

    def plot(learner, npoints):
        adaptive.runner.simple(learner, lambda l: l.npoints == npoints)
        return learner.plot().relabel(f'loss={learner.loss():.2f}')

    plots = {n: plot(learner, n) for n in range(10, 10000, 200)}
    hv.HoloMap(plots, kdims=['npoints'])

`adaptive.LearnerND`
~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    def sphere(xyz):
        import numpy as np
        x, y, z = xyz
        a = 0.4
        return np.exp(-(x**2 + y**2 + z**2 - 0.75**2)**2/a**4)

    learner = adaptive.LearnerND(sphere, bounds=[(-1, 1), (-1, 1), (-1, 1)])
    adaptive.runner.simple(learner, lambda l: l.npoints == 5000)

    fig = learner.plot_3D(return_fig=True)

    # Remove a slice from the plot to show the inside of the sphere
    scatter = fig.data[0]
    coords_col = [
        (x, y, z, color)
        for x, y, z, color in zip(
            scatter["x"], scatter["y"], scatter["z"], scatter.marker["color"]
        )
        if not (x > 0 and y > 0)
    ]
    scatter["x"], scatter["y"], scatter["z"], scatter.marker["color"] = zip(*coords_col)

    fig

see more in the :ref:`Tutorial Adaptive`.
