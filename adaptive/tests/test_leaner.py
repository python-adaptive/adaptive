# -*- coding: utf-8 -*-
import adaptive

def test_neigbors():
    learner = adaptive.learner.Learner1D(lambda x: x, bounds=(-1.0, 1.0))
    learner.add_data([-1, 1], [-1, 1])
    assert learner.neighbors == {-1: [None, 1], 1: [-1, None]}
    learner.choose_points(1, False)
    assert learner.neighbors == {-1: [None, 1], 1: [-1, None]}
    learner.choose_points(1, True)
    assert (learner.neighbors_interp ==
            {-1: [None, 0.0], 1: [0.0, None], 0.0: [-1, 1]})
