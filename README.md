# ![][logo] adaptive

[![pipeline status](https://gitlab.kwant-project.org/qt/adaptive/badges/master/pipeline.svg)](https://gitlab.kwant-project.org/qt/adaptive/pipelines)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](benchmarks)
[![coverage report](https://gitlab.kwant-project.org/qt/adaptive/badges/master/coverage.svg)](https://gitlab.kwant-project.org/qt/adaptive/commits/master)

**Tools for adaptive parallel evaluation of functions.**

Adaptive is an [open-source](LICENSE) Python library designed to make adaptive parallel function evaluation simple. With adaptive, you can adaptively sample functions by only supplying (in general) a function and its bounds, and run it on a cluster in a few lines of code. Since `adaptive` knows the problem it is solving, it can plot the data for you (even live as the data returns) without any boilerplate. 

Check out the Adaptive [example notebook `learner.ipynb`](learner.ipynb) (or run it [live on Binder](https://mybinder.org/v2/gh/python-adaptive/adaptive/master?filepath=learner.ipynb)) to see examples of how to use `adaptive`.


**WARNING: `adaptive` is still in an early alpha development stage**


## Implemented algorithms / learners
We introduce the concept of a "learner", which is an object that knows:
* the function it is trying to "learn"
* the bounds of the function domain that you are interested in
* the data that has been calculated (which could be empty at the start)

Using this information, a learner can return as many suggested points as are requested.
Adding more data to the learner means that the newly suggested points will become better and better.

The following learners are implemented
* `Learner1D` which learns a 1D function `f: ℝ → ℝ^N`
* `Learner2D` which learns a 2D function `f: ℝ^2 → ℝ^N`
* `AverageLearner` which learns a 0D function that has a source of randomness
* `IntegratorLearner` which learns the integral value of a 1D function `f: ℝ → ℝ` up to a specified tolerance
* `BalancingLearner` which takes a list of learners and suggests points of the learners that improve the loss (quality) the most


## Installation
Adaptive works with Python 3.5 and higher on Linux, Windows, or Mac, and provides optional extensions for working with the Jupyter/IPython Notebook.

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
- Christoph Groth for his Python implementation of [`CQUAD`](https://gitlab.kwant-project.org/cwg/python-cquad) which served as inspiration for the [`IntegratorLearner`](adaptive/learner/integrator_learner.py)
- Pauli Virtanen for his `AdaptiveTriSampling` script (no longer available online since SciPy Central went down) which served as inspiration for the [`Learner2D`](adaptive/learner/learner2D.py)

For general discussion, we have a [chat channel](https://chat.quantumtinkerer.tudelft.nl/external/channels/adaptive). If you find any bugs or have any feature suggestions please file a GitLab [issue](https://gitlab.kwant-project.org/qt/adaptive/issues/new?issue) or submit a [merge request](https://gitlab.kwant-project.org/qt/adaptive/merge_requests).

[logo]: /uploads/d20444093920a4a0499e165b5061d952/logo.png "adaptive logo"
