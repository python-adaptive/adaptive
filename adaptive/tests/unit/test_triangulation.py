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
    assert tri.get_simplices_attached_to_points((0, 1, 4)) == {
        (0, 1, 2),
        (0, 3, 4),
        (1, 4, 5),
    }
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


def test_circumsphere():
    from numpy import allclose
    from numpy.random import normal, uniform

    from adaptive.learner.triangulation import circumsphere, fast_norm

    def generate_random_sphere_points(dim, radius=0):
        """https://math.stackexchange.com/a/1585996"""

        vec = [None] * (dim + 1)
        center = uniform(-100, 100, dim)
        radius = uniform(1.0, 100.0) if radius == 0 else radius
        for i in range(dim + 1):
            points = normal(0, size=dim)
            x = fast_norm(points)
            points = points / x * radius
            vec[i] = tuple(points + center)

        return radius, center, vec

    for dim in range(2, 10):
        radius, center, points = generate_random_sphere_points(dim)
        circ_center, circ_radius = circumsphere(points)
        err_msg = ""
        if not allclose(circ_center, center):
            err_msg += f"Calculated center ({circ_center}) differs from true center ({center})\n"
        if not allclose(radius, circ_radius):
            err_msg += (
                f"Calculated radius {circ_radius} differs from true radius {radius}\n"
            )
        if err_msg:
            raise AssertionError(err_msg)
