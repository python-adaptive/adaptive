Tutorial `~adaptive.IntegratorLearner`
--------------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.IntegratorLearner`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()

    import holoviews as hv
    import numpy as np

This learner learns a 1D function and calculates the integral and error
of the integral with it. It is based on Pedro Gonnet’s
`implementation <https://www.academia.edu/1976055/Adaptive_quadrature_re-revisited>`__.

Let’s try the following function with cusps (that is difficult to
integrate):

.. jupyter-execute::

    def f24(x):
        return np.floor(np.exp(x))

    xs = np.linspace(0, 3, 200)
    hv.Scatter((xs, [f24(x) for x in xs]))

Just to prove that this really is a difficult to integrate function,
let’s try a familiar function integrator `scipy.integrate.quad`, which
will give us warnings that it encounters difficulties (if we run it
in a notebook.)

.. jupyter-execute::

    import scipy.integrate
    scipy.integrate.quad(f24, 0, 3)

We initialize a learner again and pass the bounds and relative tolerance
we want to reach. Then in the `~adaptive.Runner` we pass
``goal=lambda l: l.done()`` where ``learner.done()`` is ``True`` when
the relative tolerance has been reached.

.. jupyter-execute::

    from adaptive.runner import SequentialExecutor

    learner = adaptive.IntegratorLearner(f24, bounds=(0, 3), tol=1e-8)

    # We use a SequentialExecutor, which runs the function to be learned in
    # *this* process only. This means we don't pay
    # the overhead of evaluating the function in another process.
    runner = adaptive.Runner(learner, executor=SequentialExecutor(), goal=lambda l: l.done())

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

Now we could do the live plotting again, but lets just wait untill the
runner is done.

.. jupyter-execute::

    if not runner.task.done():
        raise RuntimeError('Wait for the runner to finish before executing the cells below!')

.. jupyter-execute::

    print('The integral value is {} with the corresponding error of {}'.format(learner.igral, learner.err))
    learner.plot()
