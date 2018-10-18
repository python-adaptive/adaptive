Parallelism - using multiple cores
----------------------------------

Often you will want to evaluate the function on some remote computing
resources. ``adaptive`` works out of the box with any framework that
implements a `PEP 3148 <https://www.python.org/dev/peps/pep-3148/>`__
compliant executor that returns `concurrent.futures.Future` objects.

`concurrent.futures`
~~~~~~~~~~~~~~~~~~~~

On Unix-like systems by default `adaptive.Runner` creates a
`~concurrent.futures.ProcessPoolExecutor`, but you can also pass
one explicitly e.g. to limit the number of workers:

.. code:: python

    from concurrent.futures import ProcessPoolExecutor

    executor = ProcessPoolExecutor(max_workers=4)

    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    runner = adaptive.Runner(learner, executor=executor, goal=lambda l: l.loss() < 0.05)
    runner.live_info()
    runner.live_plot(update_interval=0.1)

`ipyparallel.Client`
~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import ipyparallel

    client = ipyparallel.Client()  # You will need to start an `ipcluster` to make this work

    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    runner = adaptive.Runner(learner, executor=client, goal=lambda l: l.loss() < 0.01)
    runner.live_info()
    runner.live_plot()

`distributed.Client`
~~~~~~~~~~~~~~~~~~~~

On Windows by default `adaptive.Runner` uses a `distributed.Client`.

.. code:: python

    import distributed

    client = distributed.Client()

    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    runner = adaptive.Runner(learner, executor=client, goal=lambda l: l.loss() < 0.01)
    runner.live_info()
    runner.live_plot(update_interval=0.1)
