import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from adaptive.learner.base_learner import uses_nth_neighbors
from adaptive.learner.learnerND import LearnerND, curvature_loss_function


def ring_of_fire(xy):
    a = 0.2
    d = 0.7
    x, y = xy
    return x + math.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


def test_learnerND_inits_loss_depends_on_neighbors_correctly():
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)])
    assert learner.nth_neighbors == 0


def test_learnerND_curvature_inits_loss_depends_on_neighbors_correctly():
    loss = curvature_loss_function()
    assert loss.nth_neighbors == 1
    learner = LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
    assert learner.nth_neighbors == 1


def test_learnerND_accepts_ConvexHull_as_input():
    triangle = ConvexHull([(0, 1), (2, 0), (0, 0)])
    learner = LearnerND(ring_of_fire, bounds=triangle)
    assert learner.nth_neighbors == 0
    assert np.allclose(learner._bbox, [(0, 2), (0, 1)])


def test_learnerND_raises_if_too_many_neigbors():
    @uses_nth_neighbors(2)
    def loss(*args):
        return 0

    assert loss.nth_neighbors == 2
    with pytest.raises(NotImplementedError):
        LearnerND(ring_of_fire, bounds=[(-1, 1), (-1, 1)], loss_per_simplex=loss)
