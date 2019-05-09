Tutorial Adaptive
=================

`Adaptive <https://github.com/python-adaptive/adaptive>`__
is a package for adaptively sampling functions with support for parallel
evaluation.

This is an introductory notebook that shows some basic use cases.

``adaptive`` needs at least Python 3.6, and the following packages:

- ``scipy``
- ``sortedcontainers``

Additionally ``adaptive`` has lots of extra functionality that makes it
simple to use from Jupyter notebooks. This extra functionality depends
on the following packages

- ``ipykernel>=4.8.0``
- ``jupyter_client>=5.2.2``
- ``holoviews``
- ``bokeh``
- ``ipywidgets``

We recommend to start with the :ref:`Tutorial `~adaptive.Learner1D``.

.. note::
    Because this documentation consists of static html, the ``live_plot``
    and ``live_info`` widget is not live. Download the notebooks
    in order to see the real behaviour.

.. toctree::
    :hidden:

    tutorial.Learner1D
    tutorial.Learner2D
    tutorial.custom_loss
    tutorial.AverageLearners
    tutorial.BalancingLearner
    tutorial.DataSaver
    tutorial.IntegratorLearner
    tutorial.LearnerND
    tutorial.SKOptLearner
    tutorial.parallelism
    tutorial.advanced-topics
