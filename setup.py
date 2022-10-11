#!/usr/bin/env python3

import os
import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    print("adaptive requires Python 3.7 or above.")
    sys.exit(1)


# Loads _version.py module without importing the whole package.
def get_version_and_cmdclass(package_name):
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_name, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("adaptive")


install_requires = [
    "scipy",
    "sortedcollections >= 1.1",
    "sortedcontainers >= 2.0",
    "cloudpickle",
    "loky >= 2.9",
]
if sys.version_info < (3, 10):
    install_requires.append("typing_extensions")

extras_require = {
    "notebook": [
        "ipython",
        "ipykernel>=4.8.0",  # because https://github.com/ipython/ipykernel/issues/274 and https://github.com/ipython/ipykernel/issues/263
        "jupyter_client>=5.2.2",  # because https://github.com/jupyter/jupyter_client/pull/314
        "holoviews>=1.9.1",
        "ipywidgets",
        "bokeh",
        "pandas",
        "matplotlib",
        "plotly",
    ],
    "testing": [
        "flaky",
        "pytest",
        "pytest-cov",
        "pytest-randomly",
        "pytest-timeout",
        "pre_commit",
        "typeguard",
    ],
    "other": [
        "dill",
        "distributed",
        "ipyparallel>=6.2.5",  # because of https://github.com/ipython/ipyparallel/issues/404
        "scikit-optimize>=0.8.1",  # because of https://github.com/scikit-optimize/scikit-optimize/issues/931
        "scikit-learn<=0.24.2",  # because of https://github.com/scikit-optimize/scikit-optimize/issues/1059
        "wexpect" if os.name == "nt" else "pexpect",
    ],
}

setup(
    name="adaptive",
    description="Parallel active learning of mathematical functions",
    version=version,
    python_requires=">=3.7",
    url="https://adaptive.readthedocs.io/",
    author="Adaptive authors",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass=cmdclass,
)
