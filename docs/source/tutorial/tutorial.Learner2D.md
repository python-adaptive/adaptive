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
# Tutorial {class}`~adaptive.Learner2D`

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour.
```

```{seealso}
The complete source code of this tutorial can be found in {jupyter-download-notebook}`tutorial.Learner2D`
```

```{code-cell}
:hide-code:

import adaptive
import holoviews as hv
import numpy as np

from functools import partial
adaptive.notebook_extension()
```

Besides 1D functions, we can also learn 2D functions: $f: ℝ^2 → ℝ$.

```{code-cell}
def ring(xy, wait=True):
    import numpy as np
    from time import sleep
    from random import random

    if wait:
        sleep(random() / 10)
    x, y = xy
    a = 0.2
    return x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)


learner = adaptive.Learner2D(ring, bounds=[(-1, 1), (-1, 1)])
```

```{code-cell}
runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)
```

```{code-cell}
:hide-code:

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell}
runner.live_info()
```

```{code-cell}
def plot(learner):
    plot = learner.plot(tri_alpha=0.2)
    return (plot.Image + plot.EdgePaths.I + plot).cols(2)


runner.live_plot(plotter=plot, update_interval=0.1)
```

```{code-cell}
import itertools

# Create a learner and add data on homogeneous grid, so that we can plot it
learner2 = adaptive.Learner2D(ring, bounds=learner.bounds)
n = int(learner.npoints**0.5)
xs, ys = [np.linspace(*bounds, n) for bounds in learner.bounds]
xys = list(itertools.product(xs, ys))
learner2.tell_many(xys, map(partial(ring, wait=False), xys))

(
    learner2.plot(n).relabel("Homogeneous grid")
    + learner.plot().relabel("With adaptive")
    + learner2.plot(n, tri_alpha=0.4)
    + learner.plot(tri_alpha=0.4)
).cols(2).opts(hv.opts.EdgePaths(color="w"))
```
