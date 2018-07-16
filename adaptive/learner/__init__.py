# -*- coding: utf-8 -*-
from contextlib import suppress

from .average_learner import AverageLearner
from .base_learner import BaseLearner
from .balancing_learner import BalancingLearner
from .learner1D import Learner1D
from .learner2D import Learner2D
from .learnerND import LearnerND
from .integrator_learner import IntegratorLearner
from .data_saver import DataSaver, make_datasaver

with suppress(ImportError):
    # Only available if 'scikit-optimize' is installed
    from .skopt_learner import SKOptLearner
