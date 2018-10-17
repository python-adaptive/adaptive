Tutorial `~adaptive.LearnerND`
------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`LearnerND`

.. execute::
    :hide-code:
    :new-notebook: LearnerND

    import adaptive
    adaptive.notebook_extension()

    import holoviews as hv
    import numpy as np

    def dynamicmap_to_holomap(dm):
        # XXX: change when https://github.com/ioam/holoviews/issues/3085
        # is fixed.
        vals = {d.name: d.values for d in dm.dimensions() if d.values}
        return hv.HoloMap(dm.select(**vals))

Besides 1 and 2 dimensional functions, we can also learn N-D functions:
:math:`\ f: ℝ^N → ℝ^M, N \ge 2, M \ge 1`.

Do keep in mind the speed and
`effectiveness <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`__
of the learner drops quickly with increasing number of dimensions.

.. execute::

    # this step takes a lot of time, it will finish at about 3300 points, which can take up to 6 minutes
    def sphere(xyz):
        x, y, z = xyz
        a = 0.4
        return x + z**2 + np.exp(-(x**2 + y**2 + z**2 - 0.75**2)**2/a**4)

    learner = adaptive.LearnerND(sphere, bounds=[(-1, 1), (-1, 1), (-1, 1)])
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)

.. execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. execute::

    runner.live_info()

Let’s plot 2D slices of the 3D function

.. execute::

    def plot_cut(x, direction, learner=learner):
        cut_mapping = {'XYZ'.index(direction): x}
        return learner.plot_slice(cut_mapping, n=100)

    dm = hv.DynamicMap(plot_cut, kdims=['val', 'direction'])
    dm = dm.redim.values(val=np.linspace(-1, 1, 11), direction=list('XYZ'))

    # In a notebook one would run `dm` however we want a statically generated
    # html, so we use a HoloMap to display it here
    dynamicmap_to_holomap(dm)

Or we can plot 1D slices

.. execute::

    %%opts Path {+framewise}
    def plot_cut(x1, x2, directions, learner=learner):
        cut_mapping = {'xyz'.index(d): x for d, x in zip(directions, [x1, x2])}
        return learner.plot_slice(cut_mapping)

    dm = hv.DynamicMap(plot_cut, kdims=['v1', 'v2', 'directions'])
    dm = dm.redim.values(v1=np.linspace(-1, 1, 6),
                    v2=np.linspace(-1, 1, 6),
                    directions=['xy', 'xz', 'yz'])

    # In a notebook one would run `dm` however we want a statically generated
    # html, so we use a HoloMap to display it here
    dynamicmap_to_holomap(dm)

The plots show some wobbles while the original function was smooth, this
is a result of the fact that the learner chooses points in 3 dimensions
and the simplices are not in the same face as we try to interpolate our
lines. However, as always, when you sample more points the graph will
become gradually smoother.
