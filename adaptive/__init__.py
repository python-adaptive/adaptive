from contextlib import suppress

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

from adaptive import learner, runner, utils  # isort:skip

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

# to avoid confusion with `notebook_extension`
del notebook_integration  # type: ignore[name-defined] # noqa: F821
