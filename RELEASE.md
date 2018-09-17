# Making a Adaptive release

This document guides a contributor through creating a release of Adaptive.


## Preflight checks

The following checks should be made *before* tagging the release.


#### Check that all issues are resolved

Check that all the issues and merge requests for the appropriate
[milestone](https://gitlab.kwant-project.org/qt/adaptive/issues)
have been resolved. Any unresolved issues should have their milestone
bumped.


#### Ensure that all tests pass

For major and minor releases we will be tagging the ``master`` branch.
This should be as simple as verifying that the 
[latest CI pipeline](https://gitlab.kwant-project.org/qt/adaptive/pipelines) 
succeeded.


#### Verify that `AUTHORS.md` is up-to-date

The following command shows the number of commits per author since the last
annotated tag:
```
t=$(git describe --abbrev=0); echo Commits since $t; git shortlog -s $t..
```

## Make a release, but do not publish it yet


### Tag the release

Make an **annotated, signed** tag for the release. The tag must have the name:
```
git tag -s v<version> -m "version <version>"
```


### Build a source tarball and wheels and test it

```
rm -fr build dist
python setup.py sdist bdist_wheel
```

This creates the file `dist/adaptive-<version>.tar.gz`.  It is a good idea to unpack it 
and check that the tests run:
```
tar xzf dist/adaptive*.tar.gz
cd adaptive-*
py.test .
```

## Upload to PyPI

```
twine upload dist/*
```


## Update the [conda-forge recipe](https://github.com/conda-forge/adaptive-feedstock)

* Fork the [feedstock repo](https://github.com/conda-forge/adaptive-feedstock)
* Change the version number and sha256 in `recipe/meta.yaml` and commit to your fork
* Open a [Pull Request](https://github.com/conda-forge/adaptive-feedstock/compare)
* Type `@conda-forge-admin, please rerender` as a comment
* When the tests succeed, merge
