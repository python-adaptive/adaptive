# ![][logo] adaptive

[![Join the chat at https://gitter.im/python-adaptive/adaptive](https://badges.gitter.im/python-adaptive/adaptive.svg)](https://gitter.im/python-adaptive/adaptive?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![pipeline status](https://gitlab.kwant-project.org/qt/adaptive/badges/master/pipeline.svg)](https://gitlab.kwant-project.org/qt/adaptive/pipelines)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](benchmarks)
[![coverage report](https://gitlab.kwant-project.org/qt/adaptive/badges/master/coverage.svg)](https://gitlab.kwant-project.org/qt/adaptive/commits/master)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/python-adaptive/adaptive/master?filepath=learner.ipynb)
[![Join the chat at https://gitter.im/python-adaptive/adaptive](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/python-adaptive/adaptive?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

**Tools for adaptive parallel evaluation of functions.**

`adaptive` is an [open-source](LICENSE) Python library designed to make adaptive parallel function evaluation simple.
With `adaptive` you just supply a function with its bounds, and it will be evaluated at the "best" points in parameter space.
With just a few lines of code you can evaluate functions on a computing cluster, live-plot the data as it returns, and fine-tune the adaptive sampling algorithm.

Check out the `adaptive` [example notebook `learner.ipynb`](learner.ipynb) (or run it [live on Binder](https://mybinder.org/v2/gh/python-adaptive/adaptive/master?filepath=learner.ipynb)) to see examples of how to use `adaptive`.


**WARNING: `adaptive` is still in an early alpha development stage**


## Implemented algorithms
The core concept in `adaptive` is that of a "learner". A "learner" samples
a function at the best places in its parameter space to get maximum
"information" about the function. As it evaluates the function
at more and more points in the parameter space, it gets a better idea of where
the best places are to sample next.

Of course, what qualifies as the "best places" will depend on your application domain!
`adaptive` makes some reasonable default choices, but the details of the adaptive
sampling are completely customizable.


The following learners are implemented:
* `Learner1D`, for 1D functions `f: ℝ → ℝ^N`,
* `Learner2D`, for 2D functions `f: ℝ^2 → ℝ^N`,
* `AverageLearner`, For stochastic functions where you want to average the result over many evaluations,
* `IntegratorLearner`, for when you want to intergrate a 1D function `f: ℝ → ℝ`,
* `BalancingLearner`, for when you want to run several learners at once, selecting the "best" one each time you get more points.

In addition to the learners, `adaptive` also provides primitives for running
the sampling across several cores and even several machines, with built-in support
for [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html),
[`ipyparallel`](https://ipyparallel.readthedocs.io/en/latest/)
and [`distributed`](https://distributed.readthedocs.io/en/latest/).


## Examples
<img src="https://user-images.githubusercontent.com/6897215/35219603-a5a70e34-ff73-11e7-9a38-90749939a4a8.gif" width='20%'> </img>
<img src="https://user-images.githubusercontent.com/6897215/35219611-ac8b2122-ff73-11e7-9332-adffab64a8ce.gif" width='40%'> </img>


## Installation
`adaptive` works with Python 3.5 and higher on Linux, Windows, or Mac, and provides optional extensions for working with the Jupyter/IPython Notebook.

The recommended way to install adaptive is using `pip`:
```bash
pip install https://gitlab.kwant-project.org/qt/adaptive/repository/master/archive.zip
```


## Development

In order to not pollute the history with the output of the notebooks, please setup the git filter by executing

```bash
git config filter.nbclearoutput.clean "jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --ClearOutputPreprocessor.remove_metadata_fields='[\"deletable\", \"editable\", \"collapsed\", \"scrolled\"]' --stdin --stdout"
```
in the repository.


## Credits
We would like to give credits to the following people:
- Pedro Gonnet for his implementation of [`CQUAD`](https://www.gnu.org/software/gsl/manual/html_node/CQUAD-doubly_002dadaptive-integration.html), "Algorithm 4" as described in "Increasing the Reliability of Adaptive Quadrature Using Explicit Interpolants", P. Gonnet, ACM Transactions on Mathematical Software, 37 (3), art. no. 26, 2010.
- Christoph Groth for his Python implementation of [`CQUAD`](https://gitlab.kwant-project.org/cwg/python-cquad) which served as basis for the [`IntegratorLearner`](adaptive/learner/integrator_learner.py).
- Pauli Virtanen for his `AdaptiveTriSampling` script (no longer available online since SciPy Central went down) which served as inspiration for the [`Learner2D`](adaptive/learner/learner2D.py).

For general discussion, we have a [Gitter chat channel](https://gitter.im/python-adaptive/adaptive). If you find any bugs or have any feature suggestions please file a GitLab [issue](https://gitlab.kwant-project.org/qt/adaptive/issues/new?issue) or submit a [merge request](https://gitlab.kwant-project.org/qt/adaptive/merge_requests).

[logo]: https://gitlab.kwant-project.org/qt/adaptive/uploads/d20444093920a4a0499e165b5061d952/logo.png "adaptive logo"
