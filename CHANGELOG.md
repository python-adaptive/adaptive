# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]


## version 0.9.0

Since [0.8.0](https://github.com/python-adaptive/adaptive/tree/v0.8.0) we fixed the following [issues](https://github.com/python-adaptive/adaptive/issues):

- [#217](https://github.com/python-adaptive/adaptive/issues/217) Command-line tool
- [#211](https://github.com/python-adaptive/adaptive/issues/211) Defining inside main() in multiprocess will report error
- [#208](https://github.com/python-adaptive/adaptive/issues/208) Inquiry on implementation of parallelism on the cluster
- [#207](https://github.com/python-adaptive/adaptive/issues/207) PyYAML yaml.load(input) Deprecation
- [#203](https://github.com/python-adaptive/adaptive/issues/203) jupyter-sphinx update Documentation enhancement
- [#199](https://github.com/python-adaptive/adaptive/issues/199) jupyter-sphinx is pinned to non-existing branch

and merged the following [Pull requests](https://github.com/python-adaptive/adaptive/pulls):
- [#219](https://github.com/python-adaptive/adaptive/pull/219) pass value_scale to the LearnerND's loss_per_simplex function
- [#209](https://github.com/python-adaptive/adaptive/pull/209) remove MPI4PY_MAX_WORKERS where it's not used
- [#204](https://github.com/python-adaptive/adaptive/pull/204) use jupyter_sphinx v0.2.0 from conda instead of my branch
- [#200](https://github.com/python-adaptive/adaptive/pull/200) ensure atomic writes when saving a file
- [#193](https://github.com/python-adaptive/adaptive/pull/193) Add a SequenceLearner
- [#188](https://github.com/python-adaptive/adaptive/pull/188) BalancingLearner: add a "cycle" strategy, sampling the learners one by one
- [#202](https://github.com/python-adaptive/adaptive/pull/202) Authors
- [#201](https://github.com/python-adaptive/adaptive/pull/201) Update tutorial.parallelism.rst
- [#197](https://github.com/python-adaptive/adaptive/pull/197) Add option to display a progress bar when loading a BalancingLearner
- [#195](https://github.com/python-adaptive/adaptive/pull/195) don't treat the no data case differently in the Learner1D  Learner1D
- [#194](https://github.com/python-adaptive/adaptive/pull/194) pin everything in the docs/environment.yml file


## version 0.8.0

Since [0.7.0](https://github.com/python-adaptive/adaptive/tree/v0.7.0) we fixed the following [issues](https://github.com/python-adaptive/adaptive/issues):
* [#7](https://github.com/python-adaptive/adaptive/issues/7) suggested points lie outside of domain Learner2D
* [#39](https://github.com/python-adaptive/adaptive/issues/39) What should learners do when fed the same point twice
* [#159](https://github.com/python-adaptive/adaptive/issues/159) BalancingLearner puts all points in the first child-learner when asking for points with no data present
* [#148](https://github.com/python-adaptive/adaptive/issues/148) Loading data file with no data results in an error for the BalancingLearner
* [#145](https://github.com/python-adaptive/adaptive/issues/145) Returning np.nan breaks the 1D learner
* [#54](https://github.com/python-adaptive/adaptive/issues/54) Make learnerND datastructures immutable where possible
* [gitlab:#134](https://gitlab.kwant-project.org/qt/adaptive/issues/134) Learner1D.load throws exception when file is empty
* [#166](https://github.com/python-adaptive/adaptive/issues/166) live_plot broken with latest holoviews and bokeh
* [#156](https://github.com/python-adaptive/adaptive/issues/156) Runner errors for Python 3.7 when done
* [#159](https://github.com/python-adaptive/adaptive/issues/159) BalancingLearner puts all points in the first child-learner when asking for points with no data present
* [#171](https://github.com/python-adaptive/adaptive/issues/171) default loss of LearnerND changed?
* [#163](https://github.com/python-adaptive/adaptive/issues/163) Add a page to the documentation of papers where adaptive is used
* [#179](https://github.com/python-adaptive/adaptive/issues/179) set python_requires in setup.py
* [#175](https://github.com/python-adaptive/adaptive/issues/175) Underlying algorithm and MATLAB integration


and merged the following [Pull requests](https://github.com/python-adaptive/adaptive/pulls):
* [gitlab:!141](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/141): change the simplex_queue to a SortedKeyList
* [gitlab:!142](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/142): make methods private in the LearnerND, closes #54
* [#162](https://github.com/python-adaptive/adaptive/pull/162) test flat bands in the LearnerND
* [#161](https://github.com/python-adaptive/adaptive/pull/161) import Iterable and Sized from collections.abc
* [#160](https://github.com/python-adaptive/adaptive/pull/160) Distribute first points in a BalancingLearner
* [#153](https://github.com/python-adaptive/adaptive/pull/153) invoke conda directly in CI
* [#152](https://github.com/python-adaptive/adaptive/pull/152) fix bug in curvature_loss Learner1D bug
* [#151](https://github.com/python-adaptive/adaptive/pull/151) handle NaN losses and add a test, closes #145
* [#150](https://github.com/python-adaptive/adaptive/pull/150) fix `_get_data` for the BalancingLearner
* [#149](https://github.com/python-adaptive/adaptive/pull/149) handle empty data files when loading, closes #148
* [#147](https://github.com/python-adaptive/adaptive/pull/147) remove `_deepcopy_fix` and depend on sortedcollections >= 1.1
* [#168](https://github.com/python-adaptive/adaptive/pull/168) Temporarily fix docs
* [#167](https://github.com/python-adaptive/adaptive/pull/167) fix live_plot
* [#164](https://github.com/python-adaptive/adaptive/pull/164) do not force shutdown the executor in the cleanup
* [#172](https://github.com/python-adaptive/adaptive/issues/172) LearnerND: change the required loss to 1e-3 because the loss definition changed
* [#177](https://github.com/python-adaptive/adaptive/pull/177) use the repo code in docs execute
* [#176](https://github.com/python-adaptive/adaptive/pull/176) do not inline the HoloViews JS
* [#174](https://github.com/python-adaptive/adaptive/pull/174) add a gallery page of Adaptive uses in scientific works
* [#170](https://github.com/python-adaptive/adaptive/pull/170) Add logo to the documentation
* [#180](https://github.com/python-adaptive/adaptive/pull/180) use setup(..., python_requires='>=3.6'), closes #179
* [#182](https://github.com/python-adaptive/adaptive/pull/182) 2D: do not return points outside the bounds, closes #181 bug
* [#185](https://github.com/python-adaptive/adaptive/pull/185) Add support for neighbours in loss computation in LearnerND
* [#186](https://github.com/python-adaptive/adaptive/pull/186) renormalize the plots value axis on every update
* [#189](https://github.com/python-adaptive/adaptive/pull/189) use pytest rather than py.test
* [#190](https://github.com/python-adaptive/adaptive/pull/190) add support for mpi4py


## version 0.7.0

Since [0.6.0](https://gitlab.kwant-project.org/qt/adaptive/tree/v0.6.0) we fixed the following [issues](https://gitlab.kwant-project.org/qt/adaptive/issues):
* [#122](https://gitlab.kwant-project.org/qt/adaptive/issues/122): Remove public `fname` learner attribute
* [#119](https://gitlab.kwant-project.org/qt/adaptive/issues/119): (Learner1D) add possibility to use the direct neighbors in the loss
* [#114](https://gitlab.kwant-project.org/qt/adaptive/issues/114): (LearnerND) allow any convex hull as domain
* [#121](https://gitlab.kwant-project.org/qt/adaptive/issues/121): How to handle NaN?
* [#107](https://gitlab.kwant-project.org/qt/adaptive/issues/107): Make BaseRunner an abstract base class
* [#112](https://gitlab.kwant-project.org/qt/adaptive/issues/112): (LearnerND) add iso-surface plot feature
* [#56](https://gitlab.kwant-project.org/qt/adaptive/issues/56): Improve plotting for learners
* [#118](https://gitlab.kwant-project.org/qt/adaptive/issues/118): widgets don't show up on adaptive.readthedocs.io
* [#91](https://gitlab.kwant-project.org/qt/adaptive/issues/91): Set up documentation
* [#62](https://gitlab.kwant-project.org/qt/adaptive/issues/62): AverageLearner math domain error
* [#113](https://gitlab.kwant-project.org/qt/adaptive/issues/113): make BalancingLearner work with the live_plot
* [#111](https://gitlab.kwant-project.org/qt/adaptive/issues/111): (LearnerND) flat simplices are sometimes added on the surface of the triangulation
* [#103](https://gitlab.kwant-project.org/qt/adaptive/issues/103): (BalancingLearner) make new balancinglearner that looks at the total loss rather than loss improvement
* [#110](https://gitlab.kwant-project.org/qt/adaptive/issues/110): LearnerND triangulation incomplete
* [#127](https://gitlab.kwant-project.org/qt/adaptive/issues/127): Typo in documentation for `adaptive.learner.learner2D.uniform_loss(ip)`
* [#126](https://gitlab.kwant-project.org/qt/adaptive/issues/126): (Learner1D) improve time complexity
* [#104](https://gitlab.kwant-project.org/qt/adaptive/issues/104): Learner1D could in some situations return -inf as loss improvement, which would make balancinglearner never choose to improve
* [#128](https://gitlab.kwant-project.org/qt/adaptive/issues/128): (LearnerND) fix plotting of scaled domains
* [#78](https://gitlab.kwant-project.org/qt/adaptive/issues/78): (LearnerND) scale y-values

and merged the following [Merge Requests](https://gitlab.kwant-project.org/qt/adaptive/merge_requests):
* [!131](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/131): Resolve "(Learner1D) add possibility to use the direct neighbors in the loss"
* [!137](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/137): adhere to PEP008 by using absolute imports
* [!135](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/135): test all the different loss functions in each test
* [!133](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/133): make 'fname' a parameter to 'save' and 'load' only
* [!136](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/136): build the Dockerimage used in CI
* [!134](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/134): change resolution_loss to a factory function
* [!118](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/118): add 'save' and 'load' to the learners and periodic saving to the Runner
* [!127](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/127): Resolve "(LearnerND) allow any convex hull as domain"
* [!130](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/130): save execution time on futures inside runners
* [!111](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/111): Resolve "Make BaseRunner an abstract base class"
* [!124](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/124): Resolve "(LearnerND) add iso-surface plot feature"
* [!108](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/108): exponentially decay message frequency in live_info
* [!129](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/129): add tutorials
* [!120](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/120): add documentation
* [!125](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/125): update to the latest miniver
* [!126](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/126): add check_whitespace
* [!123](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/123): add an option to plot a HoloMap with the BalancingLearner
* [!122](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/122): implement 'npoints' strategy for the 'BalancingLearner'
* [!119](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/119): (learnerND) no more (almost) flat simplices in the triangulation
* [!109](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/109): make a BalancingLearner strategy that compares the total loss rather than loss improvement
* [!117](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/117): Cache loss and display it in the live_info widget
* [!121](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/121): 2D: add loss that minimizes the area of the triangle in 3D
* [!139](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/139): Resolve "(Learner1D) improve time complexity"
* [!140](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/140): Resolve "(LearnerND) fix plotting of scaled domains"
* [!128](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/128): LearnerND scale output values before computing loss


## version 0.6.0

Since [0.5.0](https://gitlab.kwant-project.org/qt/adaptive/tree/v0.5.0) we fixed the following [issues](https://gitlab.kwant-project.org/qt/adaptive/issues):
* [#66](https://gitlab.kwant-project.org/qt/adaptive/issues/66): (refactor) learner.tell(x, None) might be renamed to learner.tell_pending(x)
* [#92](https://gitlab.kwant-project.org/qt/adaptive/issues/92): DeprecationWarning: sorted_dict.iloc is deprecated. Use SortedDict.keys() instead.
* [#94](https://gitlab.kwant-project.org/qt/adaptive/issues/94): Learner1D breaks if right bound is added before the left bound
* [#95](https://gitlab.kwant-project.org/qt/adaptive/issues/95): Learner1D's bound check algo in self.ask doesn't take self.data or self.pending_points
* [#96](https://gitlab.kwant-project.org/qt/adaptive/issues/96): Learner1D fails when function returns a list instead of a numpy.array
* [#97](https://gitlab.kwant-project.org/qt/adaptive/issues/97): Learner1D fails when a point (x, None) is added when x already exists
* [#98](https://gitlab.kwant-project.org/qt/adaptive/issues/98): Learner1D.ask breaks when adding points in some order
* [#99](https://gitlab.kwant-project.org/qt/adaptive/issues/99): Learner1D doesn't correctly set the interpolated loss when a point is added
* [#101](https://gitlab.kwant-project.org/qt/adaptive/issues/101): How should learners handle data that is outside of the domain
* [#102](https://gitlab.kwant-project.org/qt/adaptive/issues/102): No tests for the 'BalancingLearner'
* [#105](https://gitlab.kwant-project.org/qt/adaptive/issues/105): LearnerND fails for BalancingLearner test
* [#108](https://gitlab.kwant-project.org/qt/adaptive/issues/108): (BalancingLearner) loss is cached incorrectly
* [#109](https://gitlab.kwant-project.org/qt/adaptive/issues/109): Learner2D suggests same point twice

and merged the following [Merge Requests](https://gitlab.kwant-project.org/qt/adaptive/merge_requests):
* [!93](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/93): add a release guide
* [!94](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/94): add runner.max_retries
* [!95](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/95): 1D: fix the rare case where the right boundary point exists before the left bound
* [!96](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/96): More efficient 'tell_many'
* [!97](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/97): Fix #97 and #98
* [!98](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/98): Resolve "DeprecationWarning: sorted_dict.iloc is deprecated. Use SortedDict.keys() instead."
* [!99](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/99): Resolve "Learner1D's bound check algo in self.ask doesn't take self.data or self.pending_points"
* [!100](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/100): Resolve "Learner1D doesn't correctly set the interpolated loss when a point is added"
* [!101](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/101): Resolve "Learner1D fails when function returns a list instead of a numpy.array"
* [!102](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/102): introduce 'runner.retries' and 'runner.raise_if_retries_exceeded'
* [!103](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/103): 2D: rename `learner._interp` to `learner.pending_points` as in other learners
* [!104](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/104): Make the AverageLearner only return new points ...
* [!105](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/105): move specific tests for a particular learner to separate files
* [!107](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/107): Introduce `tell_pending` which replaces `tell(x, None)`
* [!112](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/112): Resolve "LearnerND fails for BalancingLearner test"
* [!113](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/113): Resolve "(BalancingLearner) loss is cached incorrectly"
* [!114](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/114): update release guide to add a `dev` tag on top of regular tags
* [!115](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/115): Resolve "How should learners handle data that is outside of the domain"
* [!116](https://gitlab.kwant-project.org/qt/adaptive/merge_requests/116): 2D: fix #109


New features
* add `learner.tell_pending` which replaces `learner.tell(x, None)`
* add error-handling with `runner.retries` and `runner.raise_if_retries_exceeded`
* make `learner.pending_points` and `runner.pending_points` public API
* rename `learner.ask(n, add_data)` -> `learner.ask(n, tell_pending)`
* added the `overhead` method to the `BlockingRunner`


## version 0.5.0

* Introduce `LearnerND` (beta)
* Add `BalancingLearner.from_product` (see `learner.ipynb` or doc-string for useage example)
* `runner.live_info()` now shows the learner's efficiency
* `runner.task.print_stack()` now displays the full traceback
* Introduced `learner.tell_many` instead of having `learner.tell` figure out whether multiple points are added ([#59](https://gitlab.kwant-project.org/qt/adaptive/issues/59))
* Fixed a [bug](https://gitlab.kwant-project.org/qt/adaptive/issues/61) that occured when a `Learner1D` had extremely narrow features

And more bugs, see https://github.com/python-adaptive/adaptive/compare/v0.4.1...v0.5.0


## version 0.4.0

Rename `choose_points` -> `ask` and `add_point`, `add_data` -> `tell` and
* several small bug fixes
* add Jorn Hoofwijk as an author


## version 0.2.0

Release with correct licensing information
Previously Christoph Groth was not properly attributed for his contributions.

This release also contains a bugfix for Windows users


## version 0.1.0

Initial interface and algorithm proposal
