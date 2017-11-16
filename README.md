# Tools for adaptive parallel evaluation of functions
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://gitlab.kwant-project.org/qt/adaptive/)

## Development

In order to not pollute the history with the output of the notebooks, please setup the git filter by executing

```
git config filter.nbclearoutput.clean "jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --ClearOutputPreprocessor.remove_metadata_fields='[\"deletable\", \"editable\", \"collapsed\", \"scrolled\"]' --stdin --stdout"
```

in the repository.
