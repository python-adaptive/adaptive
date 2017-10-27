# -*- coding: utf-8 -*-

from ..learner import Learner1D

def test_learner1d(x):
    # Uniform sampling in absence of data
    # Pass the same data point in twice, state of learner does not change
    # Pass data for a point that was not obtained from choose_points
    # Order of calls to `add_point` in between calls to `choose_points` does not change the state
    # Expected loss improvement is never larger than total loss
    # Learner that never gets data outside of a certain domain should "perform similarly" to a learner *defined* over that certain domain
    # Check that learner performance is invariant under transformations that leave the loss function invariant (rescaling domain, codomain etc.)
    # Checking "convergence" when data points are given in a random order




def test_neighbors():
    learner = Learner1D(lambda x: x, bounds=(-1.0, 1.0))
    learner.add_data([-1, 1], [-1, 1])
    assert learner.neighbors == {-1: [None, 1], 1: [-1, None]}
    learner.choose_points(1, False)
    assert learner.neighbors == {-1: [None, 1], 1: [-1, None]}
    learner.choose_points(1, True)
    assert (learner.neighbors_interp ==
            {-1: [None, 0.0], 1: [0.0, None], 0.0: [-1, 1]})
