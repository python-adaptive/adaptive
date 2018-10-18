Tutorial `~adaptive.DataSaver`
------------------------------

.. note::
   Because this documentation consists of static html, the ``live_plot``
   and ``live_info`` widget is not live. Download the notebook
   in order to see the real behaviour.

.. seealso::
    The complete source code of this tutorial can be found in
    :jupyter-download:notebook:`tutorial.DataSaver`

.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()

If the function that you want to learn returns a value along with some
metadata, you can wrap your learner in an `adaptive.DataSaver`.

In the following example the function to be learned returns its result
and the execution time in a dictionary:

.. jupyter-execute::

    from operator import itemgetter

    def f_dict(x):
        """The function evaluation takes roughly the time we `sleep`."""
        import random
        from time import sleep

        waiting_time = random.random()
        sleep(waiting_time)
        a = 0.01
        y = x + a**2 / (a**2 + x**2)
        return {'y': y, 'waiting_time': waiting_time}

    # Create the learner with the function that returns a 'dict'
    # This learner cannot be run directly, as Learner1D does not know what to do with the 'dict'
    _learner = adaptive.Learner1D(f_dict, bounds=(-1, 1))

    # Wrapping the learner with 'adaptive.DataSaver' and tell it which key it needs to learn
    learner = adaptive.DataSaver(_learner, arg_picker=itemgetter('y'))

``learner.learner`` is the original learner, so
``learner.learner.loss()`` will call the correct loss method.

.. jupyter-execute::

    runner = adaptive.Runner(learner, goal=lambda l: l.learner.loss() < 0.1)

.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!

.. jupyter-execute::

    runner.live_info()

.. jupyter-execute::

    runner.live_plot(plotter=lambda l: l.learner.plot(), update_interval=0.1)

Now the ``DataSavingLearner`` will have an dictionary attribute
``extra_data`` that has ``x`` as key and the data that was returned by
``learner.function`` as values.

.. jupyter-execute::

    learner.extra_data
