# -*- coding: utf-8 -*-

from contextlib import suppress

from adaptive.notebook_integration import (notebook_extension, live_plot,
                                   active_plotting_tasks)

from adaptive import learner
from adaptive import runner
from adaptive import utils

from adaptive.learner import (
	BaseLearner, Learner1D, Learner2D, LearnerND,
    AverageLearner, BalancingLearner, make_datasaver,
    DataSaver, IntegratorLearner
)

with suppress(ImportError):
    # Only available if 'scikit-optimize' is installed
    from adaptive.learner import SKOptLearner

from adaptive.runner import Runner, AsyncRunner, BlockingRunner

from adaptive._version import __version__
del _version

del notebook_integration  # to avoid confusion with `notebook_extension`
