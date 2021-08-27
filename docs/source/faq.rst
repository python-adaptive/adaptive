FAQ: frequently asked questions
-------------------------------


Where can I learn more about the algorithm used?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read our `draft paper <https://gitlab.kwant-project.org/qt/adaptive-paper/builds/artifacts/master/file/paper.pdf?job=make>`_ or the source code on `GitHub <https://github.com/python-adaptive/adaptive/>`_.


How do I get the data?
~~~~~~~~~~~~~~~~~~~~~~

Check ``learner.data``.


How do I learn more than one value per point?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the `adaptive.DataSaver`.


My runner failed, how do I get the error message?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check ``runner.task.print_stack()``.


How do I get a `~adaptive.Learner2D`\'s data on a grid?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``learner.interpolated_on_grid()`` optionally with a argument ``n`` to specify the the amount of points in ``x`` and ``y``.


Why can I not use a ``lambda`` with a learner?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the `~adaptive.Runner` the learner's function is evaluated in different Python processes.
Therefore, the ``function`` needs to be serialized (pickled) and send to the other Python processes; ``lambda``\s cannot be pickled.
Instead you can probably use ``functools.partial`` to accomplish what you want to do.


How do I run multiple runners?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out `Adaptive scheduler <http://adaptive-scheduler.readthedocs.io>`_, which solves the following problem of needing to run more learners than you can run with a single runner.
It easily runs on tens of thousands of cores.


What is the difference with FEM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main difference with FEM (Finite Element Method) is that one needs to globally update the mesh at every time step.

For Adaptive, we want to be able to parallelize the function evaluation and that requires an algorithm that can quickly return a new suggested point.
This means that, to minimize the time that Adaptive spends on adding newly calculated points to the data strucute, we only want to update the data of the points that are close to the new point.


What is the difference with Bayesian optimization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indeed there are similarities between what Adaptive does and Bayesian optimization.

The choice of new points is based on the previous ones.

There is a tuneable algorithm for performing this selection, and the easiest way to formulate this algorithm is by defining a loss function.

Bayesian optimization is a perfectly fine algorithm for choosing new points within adaptive. As an experiment we have interfaced ``scikit-optimize`` and implemented a learner that just wraps it.

However there are important differences why Bayesian optimization doesn't cover all the needs.
Often our aim is to explore the function and not minimize it.
Further, Bayesian optimization is most often combined with Gaussian processes because it is then possible to compute the posteriour exactly and formulate a rigorous optimization strategy.
Unfortunately Gaussian processes are computationally expensive and won't be useful with tens of thousands of points.
Adaptive is much more simple-minded and it relies only on the local properties of the data, rather than fitting it globally.

We'd say that Bayesian modeling is good for really computationally expensive data, regular grids for really cheap data, and local adaptive algorithms are somewhere in the middle.

..  I get "``concurrent.futures.process.BrokenProcessPool``: A process in the process pool was terminated abruptly while the future was running or pending." what does it mean?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    XXX: add answer!

    What is the difference with Kriging?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    XXX: add answer!


    What is the difference with adaptive meshing in CFD or computer graphics?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    XXX: add answer!


    Can I use this to tune my hyper parameters for machine learning models?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    XXX: add answer!


    How to use Adaptive with MATLAB?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    XXX: add answer!


Missing a question that you think belongs here? Let us `know <https://github.com/python-adaptive/adaptive/issues/new>`_.
