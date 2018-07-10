from collections import defaultdict
from math import factorial

import pytest

from ..learner.triangulation import Triangulation
import numpy as np

with_dimension = pytest.mark.parametrize('dim', [1, 2, 3, 4])


def _make_triangulation(points):
    num_vertices = points.shape[1] + 1
    first_simplex, points = points[:num_vertices], points[num_vertices:]
    t = Triangulation(first_simplex)
    for p in points:
        t.add_point(p)
    return t


def _make_standard_simplex(dim):
    """Return the vertices of the standard simplex in dimension 'dim'."""
    return np.vstack((np.zeros(dim), np.eye(dim)))


def _standard_simplex_volume(dim):
    return 1 / factorial(dim)


def _simplices_are_valid(t):
    """Check that 'simplices' and 'vertex_to_simplices' are consistent."""
    vertex_to_simplices = defaultdict(set)
    for simplex in t.simplices:
        for vertex in simplex:
            vertex_to_simplices[vertex].add(simplex)
    return vertex_to_simplices == t.vertex_to_simplices


@with_dimension
def test_triangulation_of_standard_simplex_is_valid(dim):
    t = Triangulation(_make_standard_simplex(dim))
    expected_simplex = tuple(range(dim + 1))
    assert t.simplices == {expected_simplex}
    assert _simplices_are_valid(t)
    assert np.isclose(t.volume(expected_simplex),
                      _standard_simplex_volume(dim))


@with_dimension
def test_adding_point_outside_standard_simplex_is_valid(dim):
    t = Triangulation(_make_standard_simplex(dim))
    t.add_point((1.1,) * dim)

    assert _simplices_are_valid(t)
    # Check that there are only 2 simplices, and that the standard
    # simplex is one of them (it was not removed with the addition of
    # the extra point).
    assert len(t.simplices) == 2
    assert tuple(range(dim + 1)) in t.simplices

    # The first and last point belong to different simplices
    assert t.vertex_to_simplices[0] != t.vertex_to_simplices[dim + 1]
    # rest of the points are shared between the simplices
    shared_simplices = t.vertex_to_simplices[1]
    assert all(shared_simplices == t.vertex_to_simplices[v]
               for v in range(1, dim + 1))


@with_dimension
def test_adding_point_inside_standard_simplex_is_valid(dim):
    t = Triangulation(_make_standard_simplex(dim))
    first_simplex = tuple(range(dim + 1))
    t.add_point((0.1,) * dim, simplex=first_simplex)  # close to the origin
    added_point = dim + 1  # *index* of added point

    assert len(t.simplices) == dim + 1
    assert all(added_point in simplex for simplex in t.simplices)

    volume = np.sum([t.volume(s) for s in t.simplices])
    assert np.isclose(volume, _standard_simplex_volume(dim))


def test_adding_point_inside_simplex_without_providing_simplex():
    c = [(0, 0), (1, 0), (0, 1)]
    t = Triangulation(c)
    t.add_point((0.1, 0.1))
    assert t.simplices == {(0, 1, 3), (0, 2, 3), (1, 2, 3)}
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert np.isclose(volume, 0.5)


def test_adding_point_on_face():
    c = [(0, 0), (1, 0), (0, 1)]
    t = Triangulation(c)
    t.add_point((0.5, 0.5))
    assert t.simplices == {(0, 1, 3), (0, 2, 3)}
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert np.isclose(volume, 0.5)





@with_dimension
def test_triangulation_volume_is_less_than_bounding_box(dim):
    eps = 1e-8
    points = np.random.random((30, dim))  # all within the unit hypercube
    t = _make_triangulation(points)
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert volume < 1+eps


@with_dimension
def test_triangulation_is_deterministic(dim):
    points = np.random.random((30, dim))
    t1 = _make_triangulation(points)
    t2 = _make_triangulation(points)
    assert t1.simplices == t2.simplices
