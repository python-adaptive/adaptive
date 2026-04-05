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
# Tutorial {class}`~adaptive.DataSaver`

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour. [^download]
```

```{code-cell} ipython3
:tags: [hide-cell]

import adaptive

adaptive.notebook_extension()
```

If the function that you want to learn returns a value along with some metadata, you can wrap your learner in an {class}`adaptive.DataSaver`.

In the following example the function to be learned returns its result and the execution time in a dictionary:

```{code-cell} ipython3
from operator import itemgetter


def f_dict(x):
    """The function evaluation takes roughly the time we `sleep`."""
    import random
    from time import sleep

    waiting_time = random.random()
    sleep(waiting_time)
    a = 0.01
    y = x + a**2 / (a**2 + x**2)
    return {"y": y, "waiting_time": waiting_time}


# Create the learner with the function that returns a 'dict'
# This learner cannot be run directly, as Learner1D does not know what to do with the 'dict'
_learner = adaptive.Learner1D(f_dict, bounds=(-1, 1))

# Wrapping the learner with 'adaptive.DataSaver' and tell it which key it needs to learn
learner = adaptive.DataSaver(_learner, arg_picker=itemgetter("y"))
```

`learner.learner` is the original learner, so `learner.learner.loss()` will call the correct loss method.

```{code-cell} ipython3
runner = adaptive.Runner(learner, loss_goal=0.1)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
runner.live_plot(plotter=lambda lrn: lrn.learner.plot(), update_interval=0.1)
```

Now the `DataSavingLearner` will have an dictionary attribute `extra_data` that has `x` as key and the data that was returned by `learner.function` as values.

```{code-cell} ipython3
learner.extra_data
```

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.DataSaver.ipynb`** and {download}`tutorial.DataSaver.md`.
