
# ![logo](https://adaptive.readthedocs.io/en/latest/_static/logo.png) *Adaptive*: Parallel Active Learning of Mathematical Functions :brain::1234:
<!-- badges-start -->

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/python-adaptive/adaptive/main?filepath=example-notebook.ipynb)
[![Conda](https://img.shields.io/badge/install%20with-conda-green.svg)](https://anaconda.org/conda-forge/adaptive)
[![Coverage](https://img.shields.io/codecov/c/github/python-adaptive/adaptive)](https://codecov.io/gh/python-adaptive/adaptive)
[![DOI](https://img.shields.io/badge/doi-10.5281%2Fzenodo.1182437-blue.svg)](https://doi.org/10.5281/zenodo.1182437)
[![Documentation](https://readthedocs.org/projects/adaptive/badge/?version=latest)](https://adaptive.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/adaptive.svg)](https://anaconda.org/conda-forge/adaptive)
[![GitHub](https://img.shields.io/github/stars/python-adaptive/adaptive.svg?style=social)](https://github.com/python-adaptive/adaptive/stargazers)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/python-adaptive/adaptive)
[![PyPI](https://img.shields.io/pypi/v/adaptive.svg)](https://pypi.python.org/pypi/adaptive)

<!-- badges-end -->

<!-- summary-start -->

Adaptive is an open-source Python library that streamlines adaptive parallel function evaluations.
Rather than calculating all points on a dense grid, it intelligently selects the "best" points in the parameter space based on your provided function and bounds.
With minimal code, you can perform evaluations on a computing cluster, display live plots, and optimize the adaptive sampling algorithm.

Adaptive is most efficient for computations where each function evaluation takes at least ≈50ms due to the overhead of selecting potentially interesting points.

To see Adaptive in action, try the [example notebook on Binder](https://mybinder.org/v2/gh/python-adaptive/adaptive/main?filepath=example-notebook.ipynb) or explore the [tutorial on Read the Docs](https://adaptive.readthedocs.io/en/latest/tutorial/tutorial.html).

<!-- summary-end -->

<details><summary><b><u>[ToC]</u></b> 📚</summary>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [:star: Key features](#star-key-features)
- [:rocket: Example usage](#rocket-example-usage)
  - [:floppy_disk: Exporting Data](#floppy_disk-exporting-data)
- [:test_tube: Implemented Algorithms](#test_tube-implemented-algorithms)
- [:package: Installation](#package-installation)
- [:wrench: Development](#wrench-development)
- [:books: Citing](#books-citing)
- [:page_facing_up: Draft Paper](#page_facing_up-draft-paper)
- [:sparkles: Credits](#sparkles-credits)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

</details>

<!-- key-features-start -->

## :star: Key features

- 🎯 **Intelligent Adaptive Sampling**: Adaptive focuses on areas of interest within a function, ensuring better results with fewer evaluations, saving time, and computational resources.
- ⚡ **Parallel Execution**: The library leverages parallel processing for faster function evaluations, making optimal use of available computational resources.
- 📊 **Live Plotting and Info Widgets**: When working in Jupyter notebooks, Adaptive offers real-time visualization of the learning process, making it easier to monitor progress and identify areas of improvement.
- 🔧 **Customizable Loss Functions**: Adaptive supports various loss functions and allows customization, enabling users to tailor the learning process according to their specific needs.
- 📈 **Support for Multidimensional Functions**: The library can handle functions with scalar or vector outputs in one or multiple dimensions, providing flexibility for a wide range of problems.
- 🧩 **Seamless Integration**: Adaptive offers a simple and intuitive interface, making it easy to integrate with existing Python projects and workflows.
- 💾 **Flexible Data Export**: The library provides options to export learned data as NumPy arrays or Pandas DataFrames, ensuring compatibility with various data processing tools.
- 🌐 **Open-Source and Community-Driven**: Adaptive is an open-source project, encouraging contributions from the community to continuously improve and expand the library's features and capabilities.

<!-- key-features-end -->

## :rocket: Example usage

Adaptively learning a 1D function and live-plotting the process in a Jupyter notebook:

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

<img src="https://user-images.githubusercontent.com/6897215/38739170-6ac7c014-3f34-11e8-9e8f-93b3a3a3d61b.gif" width='20%'> </img> <img src="https://user-images.githubusercontent.com/6897215/35219611-ac8b2122-ff73-11e7-9332-adffab64a8ce.gif" width='40%'> </img> <img src="https://user-images.githubusercontent.com/6897215/47256441-d6d53700-d480-11e8-8224-d1cc49dbdcf5.gif" width='20%'> </img>

### :floppy_disk: Exporting Data

You can export the learned data as a NumPy array:

```python
data = learner.to_numpy()
```

If you have Pandas installed, you can also export the data as a DataFrame:

```python
df = learner.to_dataframe()
```

<!-- implemented-algorithms-start -->

## :test_tube: Implemented Algorithms

The core concept in `adaptive` is the *learner*.
A *learner* samples a function at the most interesting locations within its parameter space, allowing for optimal sampling of the function.
As the function is evaluated at more points, the learner improves its understanding of the best locations to sample next.

The definition of the "best locations" depends on your application domain.
While `adaptive` provides sensible default choices, the adaptive sampling process can be fully customized.

The following learners are implemented:

<!-- implemented-algorithms-end -->

- `Learner1D`: for 1D functions `f: ℝ → ℝ^N`,
- `Learner2D`: for 2D functions `f: ℝ^2 → ℝ^N`,
- `LearnerND`: for ND functions `f: ℝ^N → ℝ^M`,
- `AverageLearner`: for random variables, allowing averaging of results over multiple evaluations,
- `AverageLearner1D`: for stochastic 1D functions, estimating the mean value at each point,
- `IntegratorLearner`: for integrating a 1D function `f: ℝ → ℝ`,
- `BalancingLearner`: for running multiple learners simultaneously and selecting the "best" one as more points are gathered.

Meta-learners (to be used with other learners):

- `BalancingLearner`: for running several learners at once, selecting the "most optimal" one each time you get more points,
- `DataSaver`: for when your function doesn't return just a scalar or a vector.

In addition to learners, `adaptive` offers primitives for parallel sampling across multiple cores or machines, with built-in support for:
[concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html),
[mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html),
[loky](https://loky.readthedocs.io/en/stable/),
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/), and
[distributed](https://distributed.readthedocs.io/en/latest/).

<!-- rest-start -->

## :package: Installation

`adaptive` works with Python 3.7 and higher on Linux, Windows, or Mac, and provides optional extensions for working with the Jupyter/IPython Notebook.

The recommended way to install adaptive is using `conda`:

```bash
conda install -c conda-forge adaptive
```

`adaptive` is also available on PyPI:

```bash
pip install "adaptive[notebook]"
```

The `[notebook]` above will also install the optional dependencies for running `adaptive` inside a Jupyter notebook.

To use Adaptive in Jupyterlab, you need to install the following labextensions.

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @pyviz/jupyterlab_pyviz
```

## :wrench: Development

Clone the repository and run `pip install -e ".[notebook,test,other]"` to add a link to the cloned repo into your Python path:

```bash
git clone git@github.com:python-adaptive/adaptive.git
cd adaptive
pip install -e ".[notebook,test,other]"
```

We recommend using a Conda environment or a virtualenv for package management during Adaptive development.

To avoid polluting the history with notebook output, set up the git filter by running:

```bash
python ipynb_filter.py
```

in the repository.

To maintain consistent code style, we use [pre-commit](https://pre-commit.com). Install it by running:

```bash
pre-commit install
```

in the repository.

## :books: Citing

If you used Adaptive in a scientific work, please cite it as follows.

```bib
@misc{Nijholt2019,
  doi = {10.5281/zenodo.1182437},
  author = {Bas Nijholt and Joseph Weston and Jorn Hoofwijk and Anton Akhmerov},
  title = {\textit{Adaptive}: parallel active learning of mathematical functions},
  publisher = {Zenodo},
  year = {2019}
}
```

## :page_facing_up: Draft Paper

If you're interested in the scientific background and principles behind Adaptive, we recommend taking a look at the [draft paper](https://github.com/python-adaptive/paper) that is currently being written.
This paper provides a comprehensive overview of the concepts, algorithms, and applications of the Adaptive library.

## :sparkles: Credits

We would like to give credits to the following people:

- Pedro Gonnet for his implementation of [CQUAD](https://www.gnu.org/software/gsl/manual/html_node/CQUAD-doubly_002dadaptive-integration.html), “Algorithm 4” as described in “Increasing the Reliability of Adaptive Quadrature Using Explicit Interpolants”, P. Gonnet, ACM Transactions on Mathematical Software, 37 (3), art. no. 26, 2010.
- Pauli Virtanen for his `AdaptiveTriSampling` script (no longer available online since SciPy Central went down) which served as inspiration for the `adaptive.Learner2D`.

<!-- rest-end -->

For general discussion, we have a [Gitter chat channel](https://gitter.im/python-adaptive/adaptive).
If you find any bugs or have any feature suggestions please file a GitHub [issue](https://github.com/python-adaptive/adaptive/issues/new) or submit a [pull request](https://github.com/python-adaptive/adaptive/pulls).
