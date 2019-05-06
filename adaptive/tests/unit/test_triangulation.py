import numpy as np
import pytest

from adaptive.learner.triangulation import Triangulation

###################################
# Points are shaped like this:    #
#                                 #
#                    ..(5)        #
#              ..--**  /          #
#         (1)**       /           #
#        / | \       /            #
#      /   |   \    /             #
#    /     |     \ /              #
#  (2)----(0)----(4)              #
#    \     |     /                #
#      \   |   /                  #
#        \ | /                    #
#         (3)                     #
#                                 #
###################################

points = np.array([(2, 2), (2, 4), (0, 3), (2, 0), (4, 2), (5, 5)])

def test_triangulation_can_find_the_simplices():
    tri = Triangulation(points)
    assert tri.simplices == {(0, 1, 4), (0, 1, 2), (0, 2, 3), (0, 3, 4), (1, 4, 5)}


def test_triangulation_can_find_neighbors():
    tri = Triangulation(points)
    assert tri.get_simplices_attached_to_points((0, 1, 4)) == {(0, 1, 2), (0, 3, 4), (1, 4, 5)}
    assert tri.get_simplices_attached_to_points((1, 4, 5)) == {(0, 1, 4)}
    assert tri.get_simplices_attached_to_points((0, 3, 4)) == {(0, 1, 4), (0, 2, 3)}


def test_triangulation_can_find_oposing_points():
    tri = Triangulation(points)
    assert tri.get_opposing_vertices((0, 1, 4)) == (5, 3, 2)
    assert tri.get_opposing_vertices((1, 4, 5)) == (None, None, 0)
    assert tri.get_opposing_vertices((0, 1, 2)) == (None, 3, 4)
    assert tri.get_opposing_vertices((0, 2, 3)) == (None, 4, 1)
    assert tri.get_opposing_vertices((0, 3, 4)) == (None, 1, 2)


def test_triangulation_can_get_oposing_points_if_only_one_simplex_exists():
    tri = Triangulation(points[:3])
    assert tri.get_opposing_vertices((0, 1, 2)) == (None, None, None)


def test_triangulation_find_opposing_vertices_raises_if_simplex_is_invalid():
    tri = Triangulation(points)
    with pytest.raises(ValueError):
        tri.get_opposing_vertices((0, 2, 1))

    with pytest.raises(ValueError):
        tri.get_opposing_vertices((2, 3, 5))
