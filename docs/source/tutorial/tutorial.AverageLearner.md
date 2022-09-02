# Tutorial `~adaptive.AverageLearner`

:::{note}
Because this documentation consists of static html, the `live_plot`
and `live_info` widget is not live. Download the notebook
in order to see the real behaviour.
:::

:::{seealso}
The complete source code of this tutorial can be found in
{jupyter-download:notebook}`tutorial.AverageLearner`
:::

```{eval-rst}
.. jupyter-execute::
    :hide-code:

    import adaptive
    adaptive.notebook_extension()
```

The next type of learner averages a function until the uncertainty in
the average meets some condition.

This is useful for sampling a random variable. The function passed to
the learner must formally take a single parameter, which should be used
like a “seed” for the (pseudo-) random variable (although in the current
implementation the seed parameter can be ignored by the function).

```{eval-rst}
.. jupyter-execute::

    def g(n):
        import random
        from time import sleep
        sleep(random.random() / 1000)
        # Properly save and restore the RNG state
        state = random.getstate()
        random.seed(n)
        val = random.gauss(0.5, 1)
        random.setstate(state)
        return val
```

```{eval-rst}
.. jupyter-execute::

    learner = adaptive.AverageLearner(g, atol=None, rtol=0.01)
    # `loss < 1` means that we reached the `rtol` or `atol`
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 1)
```

```{eval-rst}
.. jupyter-execute::
    :hide-code:

    await runner.task  # This is not needed in a notebook environment!
```

```{eval-rst}
.. jupyter-execute::

    runner.live_info()
```

```{eval-rst}
.. jupyter-execute::

    runner.live_plot(update_interval=0.1)
```
