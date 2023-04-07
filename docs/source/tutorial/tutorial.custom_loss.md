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

# Custom adaptive logic for 1D and 2D

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour. [^download]
```

```{code-cell} ipython3
:tags: [hide-cell]

import adaptive

adaptive.notebook_extension()

# Import modules that are used in multiple cells
import numpy as np
import holoviews as hv
```

{class}`~adaptive.Learner1D` and {class}`~adaptive.Learner2D` both work on the principle of subdividing their domain into subdomains, and assigning a property to each subdomain, which we call the *loss*.
The algorithm for choosing the best place to evaluate our function is then simply *take the subdomain with the largest loss and add a point in the center, creating new subdomains around this point*.

The *loss function* that defines the loss per subdomain is the canonical place to define what regions of the domain are “interesting”.
The default loss function for {class}`~adaptive.Learner1D` and {class}`~adaptive.Learner2D` is sufficient for a wide range of common cases, but it is by no means a panacea.
For example, the default loss function will tend to get stuck on divergences.

Both the {class}`~adaptive.Learner1D` and {class}`~adaptive.Learner2D` allow you to specify a *custom loss function*.
Below we illustrate how you would go about writing your own loss function.
The documentation for {class}`~adaptive.Learner1D` and {class}`~adaptive.Learner2D` specifies the signature that your loss function needs to have in order for it to work with `adaptive`.

tl;dr, one can use the following *loss functions* that **we** already implemented:

- {class}`adaptive.learner.learner1D.default_loss`
- {class}`adaptive.learner.learner1D.uniform_loss`
- {class}`adaptive.learner.learner1D.curvature_loss_function`
- {class}`adaptive.learner.learner1D.resolution_loss_function`
- {class}`adaptive.learner.learner1D.abs_min_log_loss`
- {class}`adaptive.learner.learner2D.default_loss`
- {class}`adaptive.learner.learner2D.uniform_loss`
- {class}`adaptive.learner.learner2D.minimize_triangle_surface_loss`
- {class}`adaptive.learner.learner2D.resolution_loss_function`

Whenever a loss function has `_function` appended to its name, it is a factory function that returns the loss function with certain settings.

## Uniform sampling

Say we want to properly sample a function that contains divergences.
A simple (but naive) strategy is to *uniformly* sample the domain:

```{code-cell} ipython3
def uniform_sampling_1d(xs, ys):
    dx = xs[1] - xs[0]
    return dx


def f_divergent_1d(x):
    if x == 0:
        return np.inf
    return 1 / x**2


learner = adaptive.Learner1D(
    f_divergent_1d, (-1, 1), loss_per_interval=uniform_sampling_1d
)
runner = adaptive.BlockingRunner(learner, loss_goal=0.01)
learner.plot().select(y=(0, 10000))
```

```{code-cell} ipython3
def uniform_sampling_2d(ip):
    from adaptive.learner.learner2D import areas

    A = areas(ip)
    return np.sqrt(A)


def f_divergent_2d(xy):
    x, y = xy
    return 1 / (x**2 + y**2)


learner = adaptive.Learner2D(
    f_divergent_2d, [(-1, 1), (-1, 1)], loss_per_triangle=uniform_sampling_2d
)

# this takes a while, so use the async Runner so we know *something* is happening
runner = adaptive.Runner(
    learner, goal=lambda lrn: lrn.loss() < 0.03 or lrn.npoints > 1000
)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
def plotter(lrn):
    return lrn.plot(tri_alpha=0.3).relabel("1 / (x^2 + y^2) in log scale")


runner.live_plot(update_interval=0.2, plotter=plotter)
```

The uniform sampling strategy is a common case to benchmark against, so the 1D and 2D versions are included in `adaptive` as {class}`adaptive.learner.learner1D.uniform_loss` and {class}`adaptive.learner.learner2D.uniform_loss`.

## Doing better

Of course, using `adaptive` for uniform sampling is a bit of a waste!

Let’s see if we can do a bit better.
Below we define a loss per subdomain that scales with the degree of nonlinearity of the function (this is very similar to the default loss function for {class}`~adaptive.Learner2D`), but which is 0 for subdomains smaller than a certain area, and infinite for subdomains larger than a certain area.

A loss defined in this way means that the adaptive algorithm will first prioritise subdomains that are too large (infinite loss).
After all subdomains are appropriately small it will prioritise places where the function is very nonlinear, but will ignore subdomains that are too small (0 loss).

```{code-cell} ipython3
def resolution_loss_function(min_distance=0, max_distance=1):
    """min_distance and max_distance should be in between 0 and 1
    because the total area is normalized to 1."""

    def resolution_loss(ip):
        from adaptive.learner.learner2D import default_loss, areas

        loss = default_loss(ip)

        A = areas(ip)
        # Setting areas with a small area to zero such that they won't be chosen again
        loss[A < min_distance**2] = 0

        # Setting triangles that have a size larger than max_distance to infinite loss
        loss[A > max_distance**2] = np.inf

        return loss

    return resolution_loss


loss = resolution_loss_function(min_distance=0.01)

learner = adaptive.Learner2D(f_divergent_2d, [(-1, 1), (-1, 1)], loss_per_triangle=loss)
runner = adaptive.BlockingRunner(learner, loss_goal=0.02)
learner.plot(tri_alpha=0.3).relabel("1 / (x^2 + y^2) in log scale").opts(
    hv.opts.EdgePaths(color="w"), hv.opts.Image(logz=True, colorbar=True)
)
```

Awesome! We zoom in on the singularity, but not at the expense of sampling the rest of the domain a reasonable amount.

The above strategy is available as {class}`adaptive.learner.learner2D.resolution_loss_function`.

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.custom_loss.ipynb`** and {download}`tutorial.custom_loss.md`.
