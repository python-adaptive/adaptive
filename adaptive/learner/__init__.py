# -*- coding: utf-8 -*-

from contextlib import suppress

from adaptive.learner.average_learner import AverageLearner
from adaptive.learner.base_learner import BaseLearner
from adaptive.learner.balancing_learner import BalancingLearner
from adaptive.learner.learner1D import Learner1D
from adaptive.learner.learner2D import Learner2D
from adaptive.learner.learnerND import LearnerND
from adaptive.learner.integrator_learner import IntegratorLearner
from adaptive.learner.data_saver import DataSaver, make_datasaver

with suppress(ImportError):
    # Only available if 'scikit-optimize' is installed
    from adaptive.learner.skopt_learner import SKOptLearner
