#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys


if sys.version_info < (3, 6):
    print('adaptive requires Python 3.6 or above.')
    sys.exit(1)


# Loads _version.py module without importing the whole package.
def get_version_and_cmdclass(package_name):
    import os
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('version',
                                   os.path.join(package_name, '_version.py'))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass('adaptive')


install_requires = [
    'scipy',
    'sortedcollections',
    'sortedcontainers',
]

extras_require = {
    'notebook': [
        'ipython',
        'ipykernel>=4.8.0',  # because https://github.com/ipython/ipykernel/issues/274 and https://github.com/ipython/ipykernel/issues/263
        'jupyter_client>=5.2.2',  # because https://github.com/jupyter/jupyter_client/pull/314
        'holoviews>=1.9.1',
        'ipywidgets',
        'bokeh',
        'matplotlib',
        'plotly',
    ],
}


setup(
    name='adaptive',
    description='Adaptive parallel sampling of mathematical functions',
    version=version,
    url='https://gitlab.kwant-project.org/qt/adaptive',
    author='Adaptive authors',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages('.'),
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass=cmdclass,
)
