from contextlib import suppress

from adaptive import learner, runner, utils
from adaptive._version import __version__
from adaptive.learner import (
    AverageLearner,
    AverageLearner1D,
    BalancingLearner,
    BaseLearner,
    DataSaver,
    IntegratorLearner,
    Learner1D,
    Learner2D,
    LearnerND,
    SequenceLearner,
    make_datasaver,
)
from adaptive.notebook_integration import (
    active_plotting_tasks,
    live_plot,
    notebook_extension,
)
from adaptive.runner import AsyncRunner, BlockingRunner, Runner

__all__ = [
    "learner",
    "runner",
    "utils",
    "__version__",
    "AverageLearner",
    "BalancingLearner",
    "BaseLearner",
    "DataSaver",
    "IntegratorLearner",
    "Learner1D",
    "Learner2D",
    "LearnerND",
    "AverageLearner1D",
    "make_datasaver",
    "SequenceLearner",
    "active_plotting_tasks",
    "live_plot",
    "notebook_extension",
    "AsyncRunner",
    "BlockingRunner",
    "Runner",
]

with suppress(ImportError):
    # Only available if 'scikit-optimize' is installed
    from adaptive.learner import SKOptLearner  # noqa: F401

    __all__.append("SKOptLearner")

# to avoid confusion with `notebook_extension` and `__version__`
del _version  # noqa: F821
del notebook_integration  # noqa: F821
