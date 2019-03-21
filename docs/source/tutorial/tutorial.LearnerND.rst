Tutorial `~adaptive.LearnerND`
------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.LearnerND`

.. jupyter-execute::
    :hide-code:

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

.. jupyter-execute::

    def sphere(xyz):
        x, y, z = xyz
        a = 0.4
        return x + z**2 + np.exp(-(x**2 + y**2 + z**2 - 0.75**2)**2/a**4)

    learner = adaptive.LearnerND(sphere, bounds=[(-1, 1), (-1, 1), (-1, 1)])
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 1e-3)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

Let’s plot 2D slices of the 3D function

.. jupyter-execute::

    def plot_cut(x, direction, learner=learner):
        cut_mapping = {'XYZ'.index(direction): x}
        return learner.plot_slice(cut_mapping, n=100)

    dm = hv.DynamicMap(plot_cut, kdims=['val', 'direction'])
    dm = dm.redim.values(val=np.linspace(-1, 1, 11), direction=list('XYZ'))

    # In a notebook one would run `dm` however we want a statically generated
    # html, so we use a HoloMap to display it here
    dynamicmap_to_holomap(dm)

Or we can plot 1D slices

.. jupyter-execute::

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

Using any convex shape as domain
................................

Suppose you do not simply want to sample your function on a square (in 2D) or in
a cube (in 3D). The LearnerND supports using a `scipy.spatial.ConvexHull` as
your domain. This is best illustrated in the following example.

Suppose you would like to sample you function in a cube split in half diagonally.
You could use the following code as an example:

.. jupyter-execute::

    import scipy

    def f(xyz):
        x, y, z = xyz
        return x**4 + y**4 + z**4 - (x**2+y**2+z**2)**2

    # set the bound points, you can change this to be any shape
    b = [(-1, -1, -1),
         (-1,  1, -1),
         (-1, -1,  1),
         (-1,  1,  1),
         ( 1,  1, -1),
         ( 1, -1, -1)]

    # you have to convert the points into a scipy.spatial.ConvexHull
    hull = scipy.spatial.ConvexHull(b)

    learner = adaptive.LearnerND(f, hull)
    adaptive.BlockingRunner(learner, goal=lambda l: l.npoints > 2000)

    learner.plot_isosurface(-0.5)
