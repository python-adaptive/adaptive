---
kernelspec:
  name: python3
  display_name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.13.8
---
# Tutorial {class}`~adaptive.AverageLearner`

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour. [^download]
```

```{code-cell} ipython3
:tags: [hide-cell]

import adaptive

adaptive.notebook_extension()
```

The next type of learner averages a function until the uncertainty in the average meets some condition.

This is useful for sampling a random variable.
The function passed to the learner must formally take a single parameter, which should be used like a “seed” for the (pseudo-) random variable (although in the current implementation the seed parameter can be ignored by the function).

```{code-cell} ipython3
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

```{code-cell} ipython3
learner = adaptive.AverageLearner(g, atol=None, rtol=0.01)
# `loss < 1.0` means that we reached the `rtol` or `atol`
runner = adaptive.Runner(learner, loss_goal=1.0)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
runner.live_plot(update_interval=0.1)
```

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.AverageLearner.ipynb`** and {download}`tutorial.AverageLearner.md`.
