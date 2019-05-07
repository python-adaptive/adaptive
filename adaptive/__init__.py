# -*- coding: utf-8 -*-

from contextlib import suppress

from adaptive import learner, runner, utils
from adaptive._version import __version__
from adaptive.learner import (AverageLearner, BalancingLearner, BaseLearner,
                              DataSaver, IntegratorLearner, Learner1D,
                              Learner2D, LearnerND, make_datasaver)
from adaptive.notebook_integration import (active_plotting_tasks, live_plot,
                                           notebook_extension)
from adaptive.runner import AsyncRunner, BlockingRunner, Runner

with suppress(ImportError):
    # Only available if 'scikit-optimize' is installed
    from adaptive.learner import SKOptLearner


del _version
del notebook_integration  # to avoid confusion with `notebook_extension`
