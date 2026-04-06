import numpy as np
import scipy.spatial

from adaptive.learner import LearnerND
from adaptive.runner import BlockingRunner, replay_log, simple

from .test_learners import generate_random_parametrization, ring_of_fire


def sphere(xyz):
    x, y, z = xyz
    a = 0.4
    return x + z**2 + np.exp(-((x**2 + y**2 + z**2 - 0.75**2) ** 2) / a**4)


def test_failure_case_LearnerND():
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


def test_anisotropic_3d():
    # There was a bug where the total simplex volume would exceed the bounding
    # box volume for the anisotropic 3D learner.
    # volume for the anisotropic 3d learnerND
    # learner = adaptive.LearnerND(ring, bounds=[(-1, 1), (-1, 1)], anisotropic=True)
    learner = LearnerND(sphere, bounds=[(-1, 1), (-1, 1), (-1, 1)], anisotropic=True)

    def goal(learner):
        if learner.tri:
            sum_of_simplex_volumes = np.sum(learner.tri.volumes())
            assert sum_of_simplex_volumes < 8.00000000001
        return learner.npoints >= 1000

    BlockingRunner(learner, goal, ntasks=1)

    assert learner.npoints >= 1000


def test_interior_vs_bbox_gives_same_result():
    f = generate_random_parametrization(ring_of_fire)

    control = LearnerND(f, bounds=[(-1, 1), (-1, 1)])
    hull = scipy.spatial.ConvexHull(control._bounds_points)
    learner = LearnerND(f, bounds=hull)

    simple(control, loss_goal=0.1)
    simple(learner, loss_goal=0.1)

    assert learner.data == control.data


def test_vector_return_with_a_flat_layer():
    f = generate_random_parametrization(ring_of_fire)
    g = generate_random_parametrization(ring_of_fire)
    h1 = lambda xy: np.array([f(xy), g(xy)])  # noqa: E731
    h2 = lambda xy: np.array([f(xy), 0 * g(xy)])  # noqa: E731
    h3 = lambda xy: np.array([0 * f(xy), g(xy)])  # noqa: E731
    for function in [h1, h2, h3]:
        learner = LearnerND(function, bounds=[(-1, 1), (-1, 1)])
        simple(learner, loss_goal=0.1)
