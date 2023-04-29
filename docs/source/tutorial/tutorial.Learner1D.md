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

(TutorialLearner1D)=
# Tutorial {class}`~adaptive.Learner1D`

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

import numpy as np
```

## scalar output: `f:ℝ → ℝ`

We start with the most common use-case: sampling a 1D function `f: ℝ → ℝ`.

We will use the following function, which is a smooth (linear) background with a sharp peak at a random location:

```{code-cell} ipython3
offset = random.uniform(-0.5, 0.5)


def f(x, offset=offset, wait=True):
    from random import random
    from time import sleep

    a = 0.01
    if wait:
        sleep(random() / 10)
    return x + a**2 / (a**2 + (x - offset) ** 2)
```

We start by initializing a 1D “learner”, which will suggest points to evaluate, and adapt its suggestions as more and more points are evaluated.

```{code-cell} ipython3
learner = adaptive.Learner1D(f, bounds=(-1, 1))
```

Next we create a “runner” that will request points from the learner and evaluate ‘f’ on them.

By default on Unix-like systems the runner will evaluate the points in parallel using local processes {class}`concurrent.futures.ProcessPoolExecutor`.

On Windows systems the runner will use a {class}`loky.get_reusable_executor`.
A {class}`~concurrent.futures.ProcessPoolExecutor` cannot be used on Windows for reasons.

```{code-cell} ipython3
# The end condition is when the "loss" is less than 0.01. In the context of the
# 1D learner this means that we will resolve features in 'func' with width 0.01 or wider.
runner = adaptive.Runner(learner, loss_goal=0.01)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

When instantiated in a Jupyter notebook the runner does its job in the background and does not block the IPython kernel.
We can use this to create a plot that updates as new data arrives:

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
runner.live_plot(update_interval=0.1)
```

We can now compare the adaptive sampling to a homogeneous sampling with the same number of points:

```{code-cell} ipython3
if not runner.task.done():
    raise RuntimeError(
        "Wait for the runner to finish before executing the cells below!"
    )
```

```{code-cell} ipython3
learner2 = adaptive.Learner1D(f, bounds=learner.bounds)

xs = np.linspace(*learner.bounds, len(learner.data))
learner2.tell_many(xs, map(partial(f, wait=False), xs))

learner.plot() + learner2.plot()
```

## vector output: `f:ℝ → ℝ^N`

Sometimes you may want to learn a function with vector output:

```{code-cell} ipython3
random.seed(0)
offsets = [random.uniform(-0.8, 0.8) for _ in range(3)]

# sharp peaks at random locations in the domain


def f_levels(x, offsets=offsets):
    a = 0.01
    return np.array(
        [offset + x + a**2 / (a**2 + (x - offset) ** 2) for offset in offsets]
    )
```

`adaptive` has you covered!
The `Learner1D` can be used for such functions:

```{code-cell} ipython3
learner = adaptive.Learner1D(f_levels, bounds=(-1, 1))
runner = adaptive.Runner(
    learner, loss_goal=0.01
)  # continue until `learner.loss()<=0.01`
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

## Looking at curvature

By default `adaptive` will sample more points where the (normalized) euclidean distance between the neighboring points is large.
You may achieve better results sampling more points in regions with high curvature.
To do this, you need to tell the learner to look at the curvature by specifying `loss_per_interval`.

```{code-cell} ipython3
from adaptive.learner.learner1D import (
    curvature_loss_function,
    default_loss,
    uniform_loss,
)

curvature_loss = curvature_loss_function()
learner = adaptive.Learner1D(f, bounds=(-1, 1), loss_per_interval=curvature_loss)
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
runner.live_plot(update_interval=0.1)
```

We may see the difference of homogeneous sampling vs only one interval vs including the nearest neighboring intervals in this plot.
We will look at 100 points.

```{code-cell} ipython3
def sin_exp(x):
    from math import exp, sin

    return sin(15 * x) * exp(-(x**2) * 2)


learner_h = adaptive.Learner1D(sin_exp, (-1, 1), loss_per_interval=uniform_loss)
learner_1 = adaptive.Learner1D(sin_exp, (-1, 1), loss_per_interval=default_loss)
learner_2 = adaptive.Learner1D(sin_exp, (-1, 1), loss_per_interval=curvature_loss)

# adaptive.runner.simple is a non parallel blocking runner.
adaptive.runner.simple(learner_h, npoints_goal=100)
adaptive.runner.simple(learner_1, npoints_goal=100)
adaptive.runner.simple(learner_2, npoints_goal=100)

(
    learner_h.plot().relabel("homogeneous")
    + learner_1.plot().relabel("euclidean loss")
    + learner_2.plot().relabel("curvature loss")
).cols(2)
```

More info about using custom loss functions can be found in {ref}`Custom adaptive logic for 1D and 2D`.

## Exporting the data

We can view the raw data by looking at the dictionary `learner.data`.
Alternatively, we can view the data as NumPy array with

```{code-cell} ipython3
learner.to_numpy()
```

If Pandas is installed (optional dependency), you can also run

```{code-cell} ipython3
df = learner.to_dataframe()
df
```

and load that data into a new learner with

```{code-cell} ipython3
new_learner = adaptive.Learner1D(learner.function, (-1, 1))  # create an empty learner
new_learner.load_dataframe(df)  # load the pandas.DataFrame's data
new_learner.plot()
```

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.Learner1D.ipynb`** and {download}`tutorial.Learner1D.md`.
