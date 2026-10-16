# ![logo](https://adaptive.readthedocs.io/en/latest/_static/logo.png) *Adaptive*: Parallel Active Learning of Mathematical Functions
<!-- --8<-- [start:badges] -->

[![Tests](https://img.shields.io/github/actions/workflow/status/python-adaptive/adaptive/nox.yml?branch=main&label=tests)](https://github.com/python-adaptive/adaptive/actions/workflows/nox.yml)
[![PyPI](https://img.shields.io/pypi/v/adaptive.svg)](https://pypi.org/project/adaptive/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/adaptive.svg)](https://anaconda.org/conda-forge/adaptive)
[![Python](https://img.shields.io/pypi/pyversions/adaptive.svg)](https://pypi.org/project/adaptive/)
[![Documentation](https://img.shields.io/readthedocs/adaptive)](https://adaptive.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/codecov/c/github/python-adaptive/adaptive)](https://codecov.io/gh/python-adaptive/adaptive)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/adaptive.svg?label=conda%20downloads)](https://anaconda.org/conda-forge/adaptive)
[![DOI](https://img.shields.io/badge/doi-10.5281%2Fzenodo.1182437-blue.svg)](https://doi.org/10.5281/zenodo.1182437)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/python-adaptive/adaptive/main?filepath=example-notebook.ipynb)

<!-- --8<-- [end:badges] -->

<!-- --8<-- [start:summary] -->

`adaptive` is an open-source Python library for adaptive parallel function evaluation.
You supply a function and its bounds, and `adaptive` evaluates it at the "best" points in parameter space, instead of at *all* points of a dense grid.
With a few lines of code you can run the evaluations on a computing cluster, live-plot the data as it comes in, and fine-tune the sampling algorithm.

`adaptive` works best when each function evaluation takes *at least* ≈50 ms, because picking the next points has some overhead.
It is meant for functions of a few parameters: with more dimensions, `LearnerND` quickly becomes slower and less effective.

Try the example notebook [on Binder](https://mybinder.org/v2/gh/python-adaptive/adaptive/main?filepath=example-notebook.ipynb) or read the [tutorial](https://adaptive.readthedocs.io/en/latest/tutorial/tutorial/).

With the same number of points, adaptive sampling resolves features that a homogeneous grid misses:

<!-- --8<-- [end:summary] -->

<p align="center">
  <img src="https://raw.githubusercontent.com/python-adaptive/adaptive/main/docs/_static/readme/learner1d.gif" width="43%" alt="Learner1D: homogeneous grid versus adaptive sampling of a sharp peak">
  <img src="https://raw.githubusercontent.com/python-adaptive/adaptive/main/docs/_static/readme/learner2d.gif" width="36%" alt="Learner2D: homogeneous grid versus adaptive triangulation of a ring">
  <img src="https://raw.githubusercontent.com/python-adaptive/adaptive/main/docs/_static/readme/learnerND.gif" width="19%" alt="LearnerND: sampled points concentrate on a spherical shell">
</p>

## Example

Learning a 1D function and live-plotting the process in a Jupyter notebook:

```python
from adaptive import notebook_extension, Runner, Learner1D

notebook_extension()


def peak(x, a=0.01):
    return x + a**2 / (a**2 + x**2)


learner = Learner1D(peak, bounds=(-1, 1))
runner = Runner(learner, loss_goal=0.01)
runner.live_info()
runner.live_plot()
```

The `Runner` works in the background, so the notebook stays usable while it runs.
In a script, use `BlockingRunner`, which returns once the goal is reached:

```python
from adaptive import BlockingRunner, Learner1D


def peak(x, a=0.01):
    return x + a**2 / (a**2 + x**2)


learner = Learner1D(peak, bounds=(-1, 1))
BlockingRunner(learner, loss_goal=0.01)
data = learner.to_numpy()  # array with columns x, y
```

`learner.to_dataframe()` returns the same data as a pandas DataFrame (pandas comes with the `[notebook]` extra).

<!-- --8<-- [start:implemented-algorithms] -->

## Implemented algorithms

The core concept in `adaptive` is the *learner*.
A learner samples a function at the points in its parameter space that tell it the most about the function.
As it evaluates more points, it gets a better idea of where to sample next.

What the "best" points are depends on your application.
Each learner ranks candidate points with a *loss function*.
The default losses put more points where the function changes quickly; you can pass a built-in alternative, such as `curvature_loss_function()` or `resolution_loss_function()`, or write your own (see the [custom loss tutorial](https://adaptive.readthedocs.io/en/latest/tutorial/tutorial.custom_loss/)).

The following learners are implemented:

<!-- --8<-- [end:implemented-algorithms] -->

- `Learner1D`, for 1D functions `f: ℝ → ℝ^N`,
- `Learner2D`, for 2D functions `f: ℝ^2 → ℝ^N`,
- `LearnerND`, for ND functions `f: ℝ^N → ℝ^M`,
- `AverageLearner`, for random variables where you want to average the result over many evaluations,
- `AverageLearner1D`, for stochastic 1D functions where you want to estimate the mean value of the function at each point,
- `IntegratorLearner`, for when you want to integrate a 1D function `f: ℝ → ℝ`,
- `SequenceLearner`, for when you want to evaluate a fixed sequence of points while still using the runners and data export of `adaptive`.

Meta-learners (to be used with other learners):

- `BalancingLearner`, for when you want to run several learners at once, selecting the "best" one each time you get more points,
- `DataSaver`, for when your function doesn't just return a scalar or a vector.

In addition to the learners, `adaptive` provides primitives for running the sampling across several cores and even several machines, with built-in support for
[concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html),
[mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html),
[loky](https://loky.readthedocs.io/en/stable/),
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/), and
[distributed](https://distributed.readthedocs.io/en/latest/).

By default, both runners evaluate the function in parallel on all cores of the local machine, using loky.
To use one of the others, pass it as `executor`, for example `Runner(learner, executor=distributed.Client())`; the [parallelism tutorial](https://adaptive.readthedocs.io/en/latest/tutorial/tutorial.parallelism/) has an example for each.

<!-- --8<-- [start:rest] -->

## Installation

`adaptive` works with Python 3.11 and higher on Linux, Windows, and macOS.

Install it with `conda`:

```bash
conda install -c conda-forge adaptive
```

or from PyPI:

```bash
pip install adaptive
```

Add the `[notebook]` extra (`pip install "adaptive[notebook]"`) for live plotting and widgets in JupyterLab and Jupyter Notebook, and for `to_dataframe()`; no Jupyter extensions are needed.

### Faster triangulation (optional)

Installing the optional [adaptive-triangulation](https://github.com/python-adaptive/adaptive-triangulation) package makes `adaptive.LearnerND` significantly faster by replacing the pure-Python triangulation with a Rust implementation:

```bash
pip install "adaptive[rust]"
```

No code changes are needed. The Rust backend is detected and used automatically.
To control the selection, pass `LearnerND(..., triangulation_backend="python" | "rust" | "auto")` per learner, or set the environment variable `ADAPTIVE_TRIANGULATION_BACKEND=python` to force the pure-Python implementation globally (or to `rust` to make a missing Rust backend an error instead of a silent fallback).

## Development

Clone the repository and install it in editable mode with the optional dependencies:

```bash
git clone https://github.com/python-adaptive/adaptive.git
cd adaptive
pip install -e ".[notebook,test,other]"
```

We recommend a virtual environment, for example with [uv](https://docs.astral.sh/uv/) (`uv sync --extra notebook --extra test --extra other`) or Conda.

We use [pre-commit](https://pre-commit.com) for a consistent code style; install the hooks with

```bash
pre-commit install
```

Run the tests with `pytest`.

If you edit `example-notebook.ipynb`, install `nbconvert` and run `python ipynb_filter.py` once, so git strips the notebook output on commit.

### Building the documentation

The documentation is built with [Zensical](https://zensical.org).
The executable pages in `docs/notebooks/` are Jupyter notebooks in [jupytext's MyST Markdown format](https://jupytext.readthedocs.io/en/latest/formats-markdown.html).
`docs/build_notebooks.py` executes them (with caching) and renders them into `docs/`, after which Zensical builds the site:

```bash
uv run --group docs python docs/build_notebooks.py
uv run --group docs zensical serve  # preview on http://localhost:8000
```

The README animations are made from real learner runs with `uv run --group docs python docs/make_animations.py`, which needs `ffmpeg` and Chrome or Chromium.

## Citing

If you used `adaptive` in a scientific work, please cite it as follows.

```bib
@misc{Nijholt2019,
  doi = {10.5281/zenodo.1182437},
  author = {Bas Nijholt and Joseph Weston and Jorn Hoofwijk and Anton Akhmerov},
  title = {\textit{Adaptive}: parallel active learning of mathematical functions},
  publisher = {Zenodo},
  year = {2019}
}
```

The algorithms are described in more detail in a [draft paper](https://github.com/python-adaptive/paper).

## Credits

We would like to give credits to the following people:

- Pedro Gonnet for his implementation of [CQUAD](https://www.gnu.org/software/gsl/doc/html/integration.html#cquad-doubly-adaptive-integration), "Algorithm 4" as described in "Increasing the Reliability of Adaptive Quadrature Using Explicit Interpolants", P. Gonnet, ACM Transactions on Mathematical Software, 37 (3), art. no. 26, 2010.
- Pauli Virtanen for his `AdaptiveTriSampling` script (no longer available online since SciPy Central went down), which served as inspiration for `adaptive.Learner2D`.

<!-- --8<-- [end:rest] -->
<!-- --8<-- [start:community] -->

For general discussion, we have a [Gitter chat channel](https://gitter.im/python-adaptive/adaptive).
If you find any bugs or have any feature suggestions, please file a GitHub [issue](https://github.com/python-adaptive/adaptive/issues/new) or submit a [pull request](https://github.com/python-adaptive/adaptive/pulls).

<!-- --8<-- [end:community] -->
