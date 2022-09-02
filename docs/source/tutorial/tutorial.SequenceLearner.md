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
# Tutorial {class}`~adaptive.SequenceLearner`

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour.
```

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.SequenceLearner.ipynb`** and {download}`tutorial.SequenceLearner.md`.

```{code-cell}
---
tags: [hide-cell]
---

import adaptive

adaptive.notebook_extension()

import holoviews as hv
import numpy as np
```

This learner will learn a sequence. It simply returns the points in the provided sequence when asked.

This is useful when your problem cannot be formulated in terms of another adaptive learner, but you still want to use Adaptive's routines to run, (periodically) save, and plot.

```{code-cell}
from adaptive import SequenceLearner


def f(x):
    return int(x) ** 2


seq = np.linspace(-15, 15, 1000)
learner = SequenceLearner(f, seq)

runner = adaptive.Runner(learner, SequenceLearner.done)
# that goal is same as `lambda learner: learner.done()`
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
def plotter(learner):
    data = learner.data if learner.data else []
    return hv.Scatter(data)


runner.live_plot(plotter=plotter)
```

`learner.data` contains a dictionary that maps the index of the point of `learner.sequence` to the value at that point.

To get the values in the same order as the input sequence (`learner.sequence`) use

```{code-cell}
result = learner.result()
print(result[:10])  # print the 10 first values
```
