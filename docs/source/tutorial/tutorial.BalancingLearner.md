---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: python3
  name: python3
---

# Tutorial {class}`~adaptive.BalancingLearner`

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour. [^download]
```

```{code-cell} ipython3
:tags: [hide-cell]

import adaptive

adaptive.notebook_extension()

import random
from functools import partial

import holoviews as hv
import numpy as np
```

The balancing learner is a “meta-learner” that takes a list of learners.
When you request a point from the balancing learner, it will query all of its “children” to figure out which one will give the most improvement.

The balancing learner can for example be used to implement a poor-man’s 2D learner by using the {class}`~adaptive.Learner1D`.

```{code-cell} ipython3
def h(x, offset=0):
    a = 0.01
    return x + a**2 / (a**2 + (x - offset) ** 2)


learners = [
    adaptive.Learner1D(partial(h, offset=random.uniform(-1, 1)), bounds=(-1, 1))
    for i in range(10)
]

bal_learner = adaptive.BalancingLearner(learners)
runner = adaptive.Runner(bal_learner, loss_goal=0.01)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
def plotter(learner):
    return hv.Overlay([L.plot() for L in learner.learners])


runner.live_plot(plotter=plotter, update_interval=0.1)
```

Often one wants to create a set of `learner`s for a cartesian product of parameters.
For that particular case we’ve added a `classmethod` called {class}`~adaptive.BalancingLearner.from_product`.
See how it works below

```{code-cell} ipython3
from scipy.special import eval_jacobi


def jacobi(x, n, alpha, beta):
    return eval_jacobi(n, alpha, beta, x)


combos = {
    "n": [1, 2, 4, 8],
    "alpha": np.linspace(0, 2, 3),
    "beta": np.linspace(0, 1, 5),
}

learner = adaptive.BalancingLearner.from_product(
    jacobi, adaptive.Learner1D, {"bounds": (0, 1)}, combos
)

runner = adaptive.BlockingRunner(learner, loss_goal=0.01)

# The `cdims` will automatically be set when using `from_product`, so
# `plot()` will return a HoloMap with correctly labeled sliders.
learner.plot().overlay("beta").grid().select(y=(-1, 3))
```

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.BalancingLearner.ipynb`** and {download}`tutorial.BalancingLearner.md`.
