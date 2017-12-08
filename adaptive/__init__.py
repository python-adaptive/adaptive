# -*- coding: utf-8 -*-
from .notebook_integration import notebook_extension, live_plot

from . import learner
from . import runner

from .learner import (Learner1D, Learner2D, AverageLearner,
                      BalancingLearner, DataSaver, IntegratorLearner)
from .runner import Runner

del notebook_integration  # to avoid confusion with `notebook_extension`
