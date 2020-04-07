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

`mpi4py.futures.MPIPoolExecutor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This makes sense if you want to run a ``Learner`` on a cluster non-interactively using a job script.

For example, you create the following file called ``run_learner.py``:

.. code:: python

    from mpi4py.futures import MPIPoolExecutor

    if __name__ == "__main__":  # ← use this, see warning @ https://bit.ly/2HAk0GG

        learner = adaptive.Learner1D(f, bounds=(-1, 1))

        # load the data
        learner.load(fname)

        # run until `goal` is reached with an `MPIPoolExecutor`
        runner = adaptive.Runner(
            learner,
            executor=MPIPoolExecutor(),
            shutdown_executor=True,
            goal=lambda l: l.loss() < 0.01,
        )

        # periodically save the data (in case the job dies)
        runner.start_periodic_saving(dict(fname=fname), interval=600)

        # block until runner goal reached
        runner.ioloop.run_until_complete(runner.task)

        # save one final time before exiting
        learner.save(fname)


On your laptop/desktop you can run this script like:

.. code:: bash

    export MPI4PY_MAX_WORKERS=15
    mpiexec -n 1 python run_learner.py

Or you can pass ``max_workers=15`` programmatically when creating the `MPIPoolExecutor` instance.

Inside the job script using a job queuing system use:

.. code:: bash

    mpiexec -n 16 python -m mpi4py.futures run_learner.py

How you call MPI might depend on your specific queuing system, with SLURM for example it's:

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name adaptive-example
    #SBATCH --ntasks 100

    srun -n $SLURM_NTASKS --mpi=pmi2 ~/miniconda3/envs/py37_min/bin/python -m mpi4py.futures run_learner.py
