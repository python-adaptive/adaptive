#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import importlib
from importlib.util import module_from_spec, spec_from_file_location
from setuptools import setup, find_packages
from setuptools.command.sdist import sdist as sdist_orig
from distutils.command.build import build as build_orig


if sys.version_info < (3, 6):
    print('adaptive requires Python 3.6 or above.')
    sys.exit(1)


package_name = 'adaptive'

# Load version.py module without importing 'adaptive'
def load_version_module():
    spec = spec_from_file_location('version',
                                   '{}/version.py'.format(package_name))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

version = load_version_module()


def write_version(fname):
    # This could be a hard link, so try to delete it first.  Is there any way
    # to do this atomically together with opening?
    try:
        os.remove(fname)
    except OSError:
        pass
    with open(fname, 'w') as f:
        f.write("# This file has been created by setup.py.\n"
                "version = '{}'\n".format(version.version))


class build(build_orig):
    def run(self):
        super().run()
        write_version(os.path.join(self.build_lib, package_name,
                                   version.STATIC_VERSION_FILE))


class sdist(sdist_orig):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        write_version(os.path.join(base_dir, package_name,
                                   version.STATIC_VERSION_FILE))


install_requires = [
    'scipy',
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
    ],
}


setup(
    name='adaptive',
    description='Adaptive parallel sampling of mathematical functions',
    version=version.version,
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
    cmdclass=dict(sdist=sdist, build=build),
)
