#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='adaptive',
    description='Adaptively sample mathematical functions',
    version='0.1a',

    url='https://gitlab.kwant-project.org/qt/adaptive',
    author='Adaptive authors',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License'
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['adaptive',
              'adaptive.learner'],
    install_requires=requirements,
)
