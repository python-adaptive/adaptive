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

# Tutorial {class}`~adaptive.Learner2D`

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour. [^download]
```

```{code-cell} ipython3
:tags: [hide-cell]

from functools import partial

import holoviews as hv
import numpy as np

import adaptive

adaptive.notebook_extension()
```

Besides 1D functions, we can also learn 2D functions: $f: ℝ^2 → ℝ$.

```{code-cell} ipython3
def ring(xy, wait=True):
    from random import random
    from time import sleep

    import numpy as np

    if wait:
        sleep(random() / 10)
    x, y = xy
    a = 0.2
    return x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)


learner = adaptive.Learner2D(ring, bounds=[(-1, 1), (-1, 1)])
```

```{code-cell} ipython3
runner = adaptive.Runner(learner, loss_goal=0.01)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
def plot(learner):
    plot = learner.plot(tri_alpha=0.2)
    return (plot.Image + plot.EdgePaths.I + plot).cols(2)


runner.live_plot(plotter=plot, update_interval=0.1)
```

```{code-cell} ipython3
import itertools

# Create a learner and add data on homogeneous grid, so that we can plot it
learner2 = adaptive.Learner2D(ring, bounds=learner.bounds)
n = int(learner.npoints**0.5)
xs, ys = (np.linspace(*bounds, n) for bounds in learner.bounds)
xys = list(itertools.product(xs, ys))
learner2.tell_many(xys, map(partial(ring, wait=False), xys))

(
    learner2.plot(n).relabel("Homogeneous grid")
    + learner.plot().relabel("With adaptive")
    + learner2.plot(n, tri_alpha=0.4)
    + learner.plot(tri_alpha=0.4)
).cols(2).opts(hv.opts.EdgePaths(color="w"))
```

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.Learner2D.ipynb`** and {download}`tutorial.Learner2D.md`.
