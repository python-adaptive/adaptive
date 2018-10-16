Implemented algorithms
----------------------

The core concept in ``adaptive`` is that of a *learner*. A *learner*
samples a function at the best places in its parameter space to get
maximum “information” about the function. As it evaluates the function
at more and more points in the parameter space, it gets a better idea of
where the best places are to sample next.

Of course, what qualifies as the “best places” will depend on your
application domain! ``adaptive`` makes some reasonable default choices,
but the details of the adaptive sampling are completely customizable.

The following learners are implemented:

- `~adaptive.Learner1D`, for 1D functions ``f: ℝ → ℝ^N``,
- `~adaptive.Learner2D`, for 2D functions ``f: ℝ^2 → ℝ^N``,
- `~adaptive.LearnerND`, for ND functions ``f: ℝ^N → ℝ^M``,
- `~adaptive.AverageLearner`, For stochastic functions where you want to
  average the result over many evaluations,
- `~adaptive.IntegratorLearner`, for
  when you want to intergrate a 1D function ``f: ℝ → ℝ``,
- `~adaptive.BalancingLearner`, for when you want to run several learners at once,
  selecting the “best” one each time you get more points.

In addition to the learners, ``adaptive`` also provides primitives for
running the sampling across several cores and even several machines,
with built-in support for
`concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_,
`ipyparallel <https://ipyparallel.readthedocs.io/en/latest/>`_ and
`distributed <https://distributed.readthedocs.io/en/latest/>`_.
