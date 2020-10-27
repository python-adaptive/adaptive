import numpy as np
import pytest

try:
    from adaptive.learner.skopt_learner import SKOptLearner

    with_scikit_optimize = True
except ModuleNotFoundError:
    with_scikit_optimize = False


@pytest.mark.skipif(not with_scikit_optimize, reason="scikit-optimize is not installed")
def test_skopt_learner_runs():
    """The SKOptLearner provides very few guarantees about its
    behaviour, so we only test the most basic usage
    """

    def g(x, noise_level=0.1):
        return np.sin(5 * x) * (1 - np.tanh(x ** 2)) + np.random.randn() * noise_level

    learner = SKOptLearner(g, dimensions=[(-2.0, 2.0)])

    for _ in range(11):
        (x,), _ = learner.ask(1)
        learner.tell(x, learner.function(x))


@pytest.mark.skipif(not with_scikit_optimize, reason="scikit-optimize is not installed")
def test_skopt_learner_4D_runs():
    """The SKOptLearner provides very few guarantees about its
    behaviour, so we only test the most basic usage
    In this case we test also for 4D domain
    """

    def g(x, noise_level=0.1):
        return (
            np.sin(5 * (x[0] + x[1] + x[2] + x[3]))
            * (1 - np.tanh(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))
            + np.random.randn() * noise_level
        )

    learner = SKOptLearner(
        g, dimensions=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    )

    for _ in range(11):
        (x,), _ = learner.ask(1)
        learner.tell(x, learner.function(x))
