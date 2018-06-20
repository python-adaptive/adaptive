# -*- coding: utf-8 -*-
from .notebook_integration import (notebook_extension, live_plot,
                                   active_plotting_tasks)

from . import learner
from . import runner
from . import utils

from .learner import (Learner1D, Learner2D, AverageLearner,
                      BalancingLearner, DataSaver, IntegratorLearner)
try:
    # Only available if 'scikit-optimize' is installed
    from .learner import SKOptLearner
except ImportError:
    pass

from .runner import Runner, BlockingRunner
from . import version

__version__ = version.version

del notebook_integration  # to avoid confusion with `notebook_extension`
del version
