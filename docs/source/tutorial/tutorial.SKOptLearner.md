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
# Tutorial {class}`~adaptive.SKOptLearner`

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour. [^download]
```

```{code-cell}
---
tags: [hide-cell]
---

import adaptive

adaptive.notebook_extension()

import holoviews as hv
import numpy as np
```

We have wrapped the `Optimizer` class from [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize), to show how existing libraries can be integrated with `adaptive`.

The {class}`~adaptive.SKOptLearner` attempts to “optimize” the given function `g` (i.e. find the global minimum of `g` in the window of interest).

Here we use the same example as in the `scikit-optimize` [tutorial](https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/ask-and-tell.ipynb).
Although `SKOptLearner` can optimize functions of arbitrary dimensionality, we can only plot the learner if a 1D function is being learned.

```{code-cell}
def F(x, noise_level=0.1):
    return np.sin(5 * x) * (1 - np.tanh(x**2)) + np.random.randn() * noise_level
```

```{code-cell}
learner = adaptive.SKOptLearner(
    F,
    dimensions=[(-2.0, 2.0)],
    base_estimator="GP",
    acq_func="gp_hedge",
    acq_optimizer="lbfgs",
)
runner = adaptive.Runner(learner, ntasks=1, goal=lambda l: l.npoints > 40)
```

```{code-cell}
---
tags: [hide-cell]
---

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell}
runner.live_info()
```

```{code-cell}
xs = np.linspace(*learner.space.bounds[0])
to_learn = hv.Curve((xs, [F(x, 0) for x in xs]), label="to learn")

plot = runner.live_plot().relabel("prediction", depth=2) * to_learn
plot.opts(legend_position="top")
```

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.SKOptLearner.ipynb`** and {download}`tutorial.SKOptLearner.md`.
