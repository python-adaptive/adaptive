import numpy as np
import scipy.spatial

from adaptive.learner import LearnerND
from adaptive.runner import replay_log, simple

from .test_learners import generate_random_parametrization, ring_of_fire


def test_faiure_case_LearnerND():
    log = [
        ("ask", 4),
        ("tell", (-1, -1, -1), 1.607873907219222e-101),
        ("tell", (-1, -1, 1), 1.607873907219222e-101),
        ("ask", 2),
        ("tell", (-1, 1, -1), 1.607873907219222e-101),
        ("tell", (-1, 1, 1), 1.607873907219222e-101),
        ("ask", 2),
        ("tell", (1, -1, 1), 2.0),
        ("tell", (1, -1, -1), 2.0),
        ("ask", 2),
        ("tell", (0.0, 0.0, 0.0), 4.288304431237686e-06),
        ("tell", (1, 1, -1), 2.0),
    ]
    learner = LearnerND(lambda *x: x, bounds=[(-1, 1), (-1, 1), (-1, 1)])
    replay_log(learner, log)


def test_interior_vs_bbox_gives_same_result():
    f = generate_random_parametrization(ring_of_fire)

    control = LearnerND(f, bounds=[(-1, 1), (-1, 1)])
    hull = scipy.spatial.ConvexHull(control._bounds_points)
    learner = LearnerND(f, bounds=hull)

    simple(control, goal=lambda l: l.loss() < 0.1)
    simple(learner, goal=lambda l: l.loss() < 0.1)

    assert learner.data == control.data


def test_vector_return_with_a_flat_layer():
    f = generate_random_parametrization(ring_of_fire)
    g = generate_random_parametrization(ring_of_fire)
    h1 = lambda xy: np.array([f(xy), g(xy)])  # noqa: E731
    h2 = lambda xy: np.array([f(xy), 0 * g(xy)])  # noqa: E731
    h3 = lambda xy: np.array([0 * f(xy), g(xy)])  # noqa: E731
    for function in [h1, h2, h3]:
        learner = LearnerND(function, bounds=[(-1, 1), (-1, 1)])
        simple(learner, goal=lambda l: l.loss() < 0.1)
