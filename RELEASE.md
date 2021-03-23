# Making a Adaptive release

This document guides a contributor through creating a release of Adaptive.


## Preflight checks

The following checks should be made *before* tagging the release.


#### Check that all issues are resolved

Check that all the issues and merge requests for the appropriate
[milestone](https://github.com/python-adaptive/adaptive/issues)
have been resolved. Any unresolved issues should have their milestone
bumped.


#### Ensure that all tests pass

For major and minor releases we will be tagging the ``master`` branch.
This should be as simple as verifying that the
[latest CI pipeline](https://dev.azure.com/python-adaptive/adaptive/_build)
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

### Update the changelog
Use
```
docker run -it --rm -v "$(pwd)":/usr/local/src/your-app ferrarimarco/github-changelog-generator -u python-adaptive -p adaptive -t API_TOKEN_HERE
```
and commit the relevant parts using
```
git commit -p CHANGELOG.md
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

### Create an empty commit for new development and tag it
```
git commit --allow-empty -m 'start development towards v<version+1>'
git tag -am 'Start development towards v<version+1>' v<version+1>-dev
```

Where `<version+1>` is `<version>` with the minor version incremented
(or major version incremented and minor and patch versions then reset to 0).
This is necessary so that the reported version for any further commits is
`<version+1>-devX` and not `<version>-devX`.


## Publish the release

### Push the tags
```
git push origin v<version> v<version+1>-dev
```

### Upload to PyPI

```
twine upload dist/*
```


## Update the [conda-forge recipe](https://github.com/conda-forge/adaptive-feedstock)

* Fork the [feedstock repo](https://github.com/conda-forge/adaptive-feedstock)
* Change the version number and sha256 in `recipe/meta.yaml` and commit to your fork
* Open a [Pull Request](https://github.com/conda-forge/adaptive-feedstock/compare)
* Type `@conda-forge-admin, please rerender` as a comment
* When the tests succeed, merge
