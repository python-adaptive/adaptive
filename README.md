<p align="center">
  <img src="https://raw.githubusercontent.com/python-adaptive/adaptive/main/docs/_static/logo_docs.png" width="110" alt="Adaptive logo">
</p>

<h1 align="center">Adaptive</h1>

<p align="center"><em>Parallel active learning of mathematical functions</em></p>

<!-- --8<-- [start:badges] -->

[![Tests](https://img.shields.io/github/actions/workflow/status/python-adaptive/adaptive/nox.yml?branch=main&label=tests)](https://github.com/python-adaptive/adaptive/actions/workflows/nox.yml)
[![PyPI](https://img.shields.io/pypi/v/adaptive.svg)](https://pypi.org/project/adaptive/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/adaptive.svg)](https://anaconda.org/conda-forge/adaptive)
[![Python](https://img.shields.io/pypi/pyversions/adaptive.svg)](https://pypi.org/project/adaptive/)
[![Documentation](https://img.shields.io/readthedocs/adaptive)](https://adaptive.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/codecov/c/github/python-adaptive/adaptive)](https://codecov.io/gh/python-adaptive/adaptive)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/adaptive.svg)](https://anaconda.org/conda-forge/adaptive)
[![DOI](https://img.shields.io/badge/doi-10.5281%2Fzenodo.1182437-blue.svg)](https://doi.org/10.5281/zenodo.1182437)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/python-adaptive/adaptive/main?filepath=example-notebook.ipynb)

<!-- --8<-- [end:badges] -->

<!-- --8<-- [start:summary] -->

Adaptive samples a function where it matters.
Instead of evaluating it on a dense grid, Adaptive places each new point where its current approximation of the function is worst, so sharp features are resolved with far fewer evaluations.
The points are evaluated in parallel, on your own cores or on a cluster, and in Jupyter you can watch the result take shape while it runs.

Adaptive pays off when a single evaluation takes at least about 50 ms, as with simulations, experiments, or expensive numerics.
For cheaper functions, choosing the points costs more than evaluating them.

Try it in your browser [on Binder](https://mybinder.org/v2/gh/python-adaptive/adaptive/main?filepath=example-notebook.ipynb), or start with the [tutorial](https://adaptive.readthedocs.io/en/latest/tutorial/tutorial/).

<!-- --8<-- [end:summary] -->

<p align="center">
  <img src="https://raw.githubusercontent.com/python-adaptive/adaptive/main/docs/_static/readme/learner1d.gif" width="43%" alt="Learner1D: homogeneous grid versus adaptive sampling of a sharp peak">
  <img src="https://raw.githubusercontent.com/python-adaptive/adaptive/main/docs/_static/readme/learner2d.gif" width="36%" alt="Learner2D: homogeneous grid versus adaptive triangulation of a ring">
  <img src="https://raw.githubusercontent.com/python-adaptive/adaptive/main/docs/_static/readme/learnerND.gif" width="19%" alt="LearnerND: sampled points concentrate on a spherical shell">
</p>
<p align="center"><em>The same number of points on a homogeneous grid and placed by Adaptive.</em></p>

<!-- --8<-- [start:key-features] -->

## Features

- **Fewer evaluations**: points go where the function changes, not where a grid puts them.
- **Parallel**: runners use all local cores by default, and scale to a cluster with any `concurrent.futures`-style executor, such as ipyparallel, Dask's `distributed`, or mpi4py.
- **Live in Jupyter**: the runner works in the background while `live_plot()` and `live_info()` show its progress, so the notebook stays usable.
- **Customizable**: stop on a loss, a number of points, or a time limit, and replace the default loss to define what "interesting" means for your problem.
- **Your data, your way**: save and resume runs with `learner.save()` and `learner.load()`, and export to NumPy or pandas with `to_numpy()` and `to_dataframe()`.

<!-- --8<-- [end:key-features] -->

## Quick start

```bash
pip install adaptive
```

Learn a function with a sharp peak, evaluating it in parallel until the loss is below 0.01:

```python
import adaptive


def peak(x, a=0.01):
    return x + a**2 / (a**2 + x**2)


learner = adaptive.Learner1D(peak, bounds=(-1, 1))
adaptive.BlockingRunner(learner, loss_goal=0.01)

data = learner.to_numpy()  # (x, y) rows; or learner.to_dataframe() with pandas
```

In Jupyter (`pip install "adaptive[notebook]"`), use the non-blocking `Runner` and watch it work:

```python
adaptive.notebook_extension()

learner = adaptive.Learner1D(peak, bounds=(-1, 1))
runner = adaptive.Runner(learner, loss_goal=0.01)
runner.live_info()  # progress, and a button to stop the runner
runner.live_plot()  # updates as new points come in
```

<!-- --8<-- [start:implemented-algorithms] -->

## Learners

A *learner* decides where to sample next, and a *runner* evaluates those points and feeds the results back until a goal is reached.
Which point is "best" depends on the problem: every learner has a sensible default, and you can replace its loss function with your own.

Adaptive comes with these learners:

<!-- --8<-- [end:implemented-algorithms] -->

| Learner             | For                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------- |
| `Learner1D`         | 1D functions `f: ℝ → ℝ^N`                                                              |
| `Learner2D`         | 2D functions `f: ℝ^2 → ℝ^N`                                                            |
| `LearnerND`         | ND functions `f: ℝ^N → ℝ^M`                                                            |
| `AverageLearner`    | random variables, averaged over many evaluations                                       |
| `AverageLearner1D`  | noisy 1D functions, estimating the mean at each point                                  |
| `IntegratorLearner` | the integral of a 1D function `f: ℝ → ℝ`                                               |
| `SequenceLearner`   | a fixed sequence of points, with the runners and data export of Adaptive               |
| `BalancingLearner`  | several learners at once, each new point going to the learner that needs it most       |
| `DataSaver`         | wrapping another learner when the function returns more than a scalar or a vector      |

Runners accept any executor, so the same code runs on a laptop or a cluster, with built-in support for
[concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html),
[loky](https://loky.readthedocs.io/en/stable/) (the default),
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/),
[distributed](https://distributed.readthedocs.io/en/latest/), and
[mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html):

```python
from distributed import Client

runner = adaptive.Runner(learner, executor=Client(), loss_goal=0.01)
```

## Documentation

The [documentation](https://adaptive.readthedocs.io/) has a [tutorial](https://adaptive.readthedocs.io/en/latest/tutorial/tutorial/) for each learner, [interactive examples](https://adaptive.readthedocs.io/en/latest/algorithms_and_examples/) comparing adaptive and homogeneous sampling, [benchmarks](https://adaptive.readthedocs.io/en/latest/benchmarks/), a [gallery](https://adaptive.readthedocs.io/en/latest/gallery/) of projects that use Adaptive, and the [API reference](https://adaptive.readthedocs.io/en/latest/reference/adaptive/).

<!-- --8<-- [start:rest] -->

## Installation

Adaptive needs Python 3.11 or newer and runs on Linux, macOS, and Windows.
Install it from PyPI or conda-forge:

```bash
pip install adaptive
conda install -c conda-forge adaptive
```

Optional extras:

| Extra                | Adds                                                                                    |
| -------------------- | --------------------------------------------------------------------------------------- |
| `adaptive[notebook]` | live plots and widgets in JupyterLab and Jupyter Notebook; no Jupyter extensions needed |
| `adaptive[rust]`     | [adaptive-triangulation](https://github.com/python-adaptive/adaptive-triangulation), a faster Rust triangulation for `LearnerND` |
| `adaptive[other]`    | ipyparallel and distributed executors, and dill for serialization                       |

The Rust triangulation is used automatically when it is installed.
To choose explicitly, pass `LearnerND(..., triangulation_backend="python" | "rust" | "auto")`, or set the environment variable `ADAPTIVE_TRIANGULATION_BACKEND`; with `rust`, a missing backend is an error instead of a silent fallback.

## Development

Clone the repository and install it in editable mode, for example with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/python-adaptive/adaptive.git
cd adaptive
uv sync --extra notebook --extra test --extra other
```

or with pip in a virtual environment: `pip install -e ".[notebook,test,other]"`.

Then set up the git filter that strips notebook outputs, and [pre-commit](https://pre-commit.com) for the code style:

```bash
python ipynb_filter.py
pre-commit install
```

### Building the documentation

The documentation is built with [Zensical](https://zensical.org).
The executable pages in `docs/notebooks/` are Jupyter notebooks in [jupytext's MyST Markdown format](https://jupytext.readthedocs.io/en/latest/formats-markdown.html).
`docs/build_notebooks.py` executes them (with caching) and renders them into `docs/`, after which Zensical builds the site:

```bash
uv run --group docs python docs/build_notebooks.py
uv run --group docs zensical serve  # preview on http://localhost:8000
```

The README animations are rendered from real learner runs with `uv run --group docs python docs/make_animations.py`, which needs `ffmpeg` and Chrome or Chromium.

## Citing

If you use Adaptive in scientific work, please cite it:

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

- Pedro Gonnet for his implementation of [CQUAD](https://www.gnu.org/software/gsl/manual/html_node/CQUAD-doubly_002dadaptive-integration.html), "Algorithm 4" as described in "Increasing the Reliability of Adaptive Quadrature Using Explicit Interpolants", P. Gonnet, ACM Transactions on Mathematical Software, 37 (3), art. no. 26, 2010.
- Pauli Virtanen for his `AdaptiveTriSampling` script (no longer available online since SciPy Central went down), which inspired `adaptive.Learner2D`.

<!-- --8<-- [end:rest] -->
<!-- --8<-- [start:community] -->

## Community

For general discussion, there is a [Gitter chat channel](https://gitter.im/python-adaptive/adaptive).
Found a bug or have an idea? Open an [issue](https://github.com/python-adaptive/adaptive/issues/new) or a [pull request](https://github.com/python-adaptive/adaptive/pulls).

<!-- --8<-- [end:community] -->
