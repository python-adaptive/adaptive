from contextlib import suppress

from adaptive.learner.average_learner import AverageLearner
from adaptive.learner.average_learner1D import AverageLearner1D
from adaptive.learner.balancing_learner import BalancingLearner
from adaptive.learner.base_learner import BaseLearner
from adaptive.learner.data_saver import DataSaver, make_datasaver
from adaptive.learner.integrator_learner import IntegratorLearner
from adaptive.learner.learner1D import Learner1D
from adaptive.learner.learner2D import Learner2D
from adaptive.learner.learnerND import LearnerND
from adaptive.learner.sequence_learner import SequenceLearner

__all__ = [
    "AverageLearner",
    "BalancingLearner",
    "BaseLearner",
    "DataSaver",
    "make_datasaver",
    "IntegratorLearner",
    "Learner1D",
    "Learner2D",
    "LearnerND",
    "AverageLearner1D",
    "SequenceLearner",
]

with suppress(ImportError):
    # Only available if 'scikit-optimize' is installed
    from adaptive.learner.skopt_learner import SKOptLearner  # noqa: F401

    __all__.append("SKOptLearner")
