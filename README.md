# ![][logo] adaptive

[![](https://gitlab.kwant-project.org/qt/adaptive/badges/master/build.svg)](https://gitlab.kwant-project.org/qt/adaptive/pipelines)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://gitlab.kwant-project.org/qt/adaptive/)

**Tools for adaptive parallel evaluation of functions.**

Adaptive is an [open-source](LICENSE) Python library designed to make adaptive parallel function evaluation simple. With adaptive, you can adaptively sample functions by only supplying (in general) a function and its bounds, and run it on a cluster in a few lines of code. Since `adaptive` knows the problem it is solving, it can plot the data for you (even live as the data returns) without any boilerplate. 

Check out the Adaptive [example notebook `learner.ipynb`](learner.ipynb) (or run it [live on Binder](https://mybinder.org/v2/gh/python-adaptive/adaptive/master?filepath=learner.ipynb)) to see examples of how to use `adaptive`.


**WARNING: `adaptive` is still in an early alpha development stage**


## Installation
Adaptive works with Python 3.5 and higher on Linux, Windows, or Mac, and provides optional extensions for working with the Jupyter/IPython Notebook.

The recommended way to install adaptive is using the pip:
```
pip install https://gitlab.kwant-project.org/qt/adaptive/repository/master/archive.zip
```


## Development

In order to not pollute the history with the output of the notebooks, please setup the git filter by executing

```
git config filter.nbclearoutput.clean "jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --ClearOutputPreprocessor.remove_metadata_fields='[\"deletable\", \"editable\", \"collapsed\", \"scrolled\"]' --stdin --stdout"
```
in the repository.


For general discussion, we have a [chat channel](https://chat.quantumtinkerer.tudelft.nl/external/channels/adaptive). If you find any bugs or have any feature suggestions please file a GitLab [issue](https://gitlab.kwant-project.org/qt/adaptive/issues/new?issue) or submit a [merge request](https://gitlab.kwant-project.org/qt/adaptive/merge_requests).

[logo]: /uploads/d20444093920a4a0499e165b5061d952/logo.png "adaptive logo"
