# 📜 API reference

## Learners

| Page | Description |
| --- | --- |
| [`Learner1D`](adaptive.learner.learner1D.md) | 1D functions `f: ℝ → ℝ^N`, and its loss functions |
| [`Learner2D`](adaptive.learner.learner2D.md) | 2D functions `f: ℝ^2 → ℝ^N`, and its loss functions |
| [`LearnerND`](adaptive.learner.learnerND.md) | ND functions `f: ℝ^N → ℝ^M`, and its loss functions |
| [`AverageLearner`](adaptive.learner.average_learner.md) | Averaging a random variable |
| [`AverageLearner1D`](adaptive.learner.average_learner1D.md) | Stochastic 1D functions |
| [`BalancingLearner`](adaptive.learner.balancing_learner.md) | Running several learners at once |
| [`DataSaver`](adaptive.learner.data_saver.md) | Functions that return more than a scalar or vector |
| [`IntegratorLearner`](adaptive.learner.integrator_learner.md) | Integrating 1D functions |
| [`SequenceLearner`](adaptive.learner.sequence_learner.md) | Evaluating a fixed sequence of points |
| [`BaseLearner`](adaptive.learner.base_learner.md) | Base class for all learners |

## Runners

| Page | Description |
| --- | --- |
| [`Runner`](adaptive.runner.Runner.md) | Alias of `AsyncRunner` |
| [`AsyncRunner`](adaptive.runner.AsyncRunner.md) | Runs a learner in the background on an asyncio event loop |
| [`BlockingRunner`](adaptive.runner.BlockingRunner.md) | Runs a learner and blocks until done |
| [`BaseRunner`](adaptive.runner.BaseRunner.md) | Base class for all runners |
| [Runner extras](adaptive.runner.extras.md) | Stopping criteria, `simple`, `SequentialExecutor`, and `replay_log` |

## Other

| Page | Description |
| --- | --- |
| [`adaptive.learner.triangulation`](adaptive.learner.triangulation.md) | Triangulation used by `LearnerND` |
| [`adaptive.learner.triangulation_backend`](adaptive.learner.triangulation_backend.md) | Selecting the Python or Rust triangulation for `LearnerND` |
| [`adaptive.notebook_integration`](adaptive.notebook_integration.md) | Live plotting and widgets in Jupyter |
| [`adaptive.utils`](adaptive.utils.md) | Utilities |
