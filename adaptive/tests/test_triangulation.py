import itertools
from collections import Counter
from math import factorial

import numpy as np
import pytest

from adaptive.learner.triangulation import Triangulation

with_dimension = pytest.mark.parametrize("dim", [2, 3, 4])


def _make_triangulation(points):
    num_vertices = points.shape[1] + 1
    first_simplex, points = points[:num_vertices], points[num_vertices:]
    t = Triangulation(first_simplex)
    for p in points:
        _add_point_with_check(t, p)
    return t


def _make_standard_simplex(dim):
    """Return the vertices of the standard simplex in dimension 'dim'."""
    return np.vstack((np.zeros(dim), np.eye(dim)))


def _standard_simplex_volume(dim):
    return 1 / factorial(dim)


def _check_simplices_are_valid(t):
    """Check that 'simplices' and 'vertex_to_simplices' are consistent."""
    vertex_to_simplices = [set() for _ in t.vertices]

    for simplex in t.simplices:
        for vertex in simplex:
            vertex_to_simplices[vertex].add(simplex)
    assert vertex_to_simplices == t.vertex_to_simplices


def _check_faces_are_valid(t):
    """Check that a 'dim-1'-D face is shared by no more than 2 simplices."""
    counts = Counter(t.faces())
    assert not any(i > 2 for i in counts.values()), counts


def _check_hull_is_valid(t):
    """Check that the stored hull is consistent with one computed from scratch."""
    counts = Counter(t.faces())
    hull = {point for face, count in counts.items() if count == 1 for point in face}
    assert t.hull == hull


def _check_triangulation_is_valid(t):
    _check_simplices_are_valid(t)
    _check_faces_are_valid(t)
    _check_hull_is_valid(t)


def _add_point_with_check(tri, point, simplex=None):
    """Check that the difference in simplices before and after adding a point
    is returned by tri.add_point"""
    old_simplices = tri.simplices.copy()
    deleted_simplices, created_simplices = tri.add_point(point, simplex=simplex)
    new_simplices = tri.simplices.copy()

    assert deleted_simplices == old_simplices - new_simplices
    assert created_simplices == new_simplices - old_simplices


def test_triangulation_raises_exception_for_1d_list():
    # We could support 1d, but we don't for now, because it is not relevant
    # so a user has to be aware
    pts = [0, 1]
    with pytest.raises(TypeError):
        Triangulation(pts)


def test_triangulation_raises_exception_for_1d_points():
    # We could support 1d, but we don't for now, because it is not relevant
    # so a user has to be aware
    pts = [(0,), (1,)]
    with pytest.raises(ValueError):
        Triangulation(pts)


@with_dimension
def test_triangulation_of_standard_simplex(dim):
    t = Triangulation(_make_standard_simplex(dim))
    expected_simplex = tuple(range(dim + 1))
    assert t.simplices == {expected_simplex}
    _check_triangulation_is_valid(t)
    assert np.isclose(t.volume(expected_simplex), _standard_simplex_volume(dim))


@with_dimension
def test_zero_volume_initial_simplex_raises_exception(dim):
    points = _make_standard_simplex(dim)[:-1]
    linearly_dependent_point = np.dot(np.random.random(dim), points)
    zero_volume_simplex = np.vstack((points, linearly_dependent_point))

    assert np.isclose(np.linalg.det(zero_volume_simplex[1:]), 0)  # sanity check

    with pytest.raises(ValueError):
        Triangulation(zero_volume_simplex)


@with_dimension
def test_adding_point_outside_circumscribed_hypersphere_in_positive_orthant(dim):
    t = Triangulation(_make_standard_simplex(dim))

    point_outside_circumscribed_sphere = (1.1,) * dim
    _add_point_with_check(t, point_outside_circumscribed_sphere)

    simplex1 = tuple(range(dim + 1))
    simplex2 = tuple(range(1, dim + 2))
    n_vertices = len(t.vertices)

    _check_triangulation_is_valid(t)
    assert t.simplices == {simplex1, simplex2}

    # All points are in the hull
    assert t.hull == set(range(n_vertices))

    assert t.vertex_to_simplices[0] == {simplex1}
    assert t.vertex_to_simplices[n_vertices - 1] == {simplex2}

    # rest of the points are shared between the 2 simplices
    shared_simplices = {simplex1, simplex2}
    assert all(
        t.vertex_to_simplices[v] == shared_simplices for v in range(1, n_vertices - 1)
    )


@with_dimension
def test_adding_point_outside_standard_simplex_in_negative_orthant(dim):
    t = Triangulation(_make_standard_simplex(dim))
    new_point = list(range(-dim, 0))

    _add_point_with_check(t, new_point)

    n_vertices = len(t.vertices)

    initial_simplex = tuple(range(dim + 1))

    _check_triangulation_is_valid(t)
    assert len(t.simplices) == dim + 1
    assert initial_simplex in t.simplices

    # Hull consists of all points except the origin
    assert set(range(1, n_vertices)) == t.hull

    # Origin belongs to all the simplices
    assert t.vertex_to_simplices[0] == t.simplices

    # new point belongs to all the simplices *except* the initial one
    assert t.vertex_to_simplices[dim + 1] == t.simplices - {initial_simplex}

    other_points = list(range(1, dim + 1))
    last_vertex = n_vertices - 1
    extra_simplices = {
        (0, *points, last_vertex)
        for points in itertools.combinations(other_points, dim - 1)
    }

    assert extra_simplices | {initial_simplex} == t.simplices


@with_dimension
@pytest.mark.parametrize("provide_simplex", [True, False])
def test_adding_point_inside_standard_simplex(dim, provide_simplex):
    t = Triangulation(_make_standard_simplex(dim))
    first_simplex = tuple(range(dim + 1))
    inside_simplex = (0.1,) * dim

    if provide_simplex:
        _add_point_with_check(t, inside_simplex, simplex=first_simplex)
    else:
        _add_point_with_check(t, inside_simplex)

    added_point = dim + 1  # *index* of added point

    _check_triangulation_is_valid(t)

    other_points = list(range(dim + 1))
    expected_simplices = {
        (*points, added_point) for points in itertools.combinations(other_points, dim)
    }
    assert expected_simplices == t.simplices

    assert np.isclose(np.sum(t.volumes()), _standard_simplex_volume(dim))


@with_dimension
def test_adding_point_on_standard_simplex_face(dim):
    pts = _make_standard_simplex(dim)
    t = Triangulation(pts)
    on_simplex = np.average(pts[1:], axis=0)

    _add_point_with_check(t, on_simplex)
    added_point = dim + 1  # *index* of added point

    _check_triangulation_is_valid(t)

    other_points = list(range(1, dim + 1))
    expected_simplices = {
        (0, *points, added_point)
        for points in itertools.combinations(other_points, dim - 1)
    }
    assert expected_simplices == t.simplices

    assert np.isclose(np.sum(t.volumes()), _standard_simplex_volume(dim))


@with_dimension
def test_adding_point_on_standard_simplex_edge(dim):
    pts = _make_standard_simplex(dim)
    t = Triangulation(pts)
    on_edge = np.average(pts[:2], axis=0)

    _add_point_with_check(t, on_edge)
    _check_triangulation_is_valid(t)

    other_points = list(range(2, dim + 2))

    new_simplices = {(0, *other_points), (1, *other_points)}

    assert new_simplices == t.simplices

    assert np.isclose(np.sum(t.volumes()), _standard_simplex_volume(dim))


@with_dimension
def test_adding_point_colinear_with_first_edge(dim):
    pts = _make_standard_simplex(dim)
    t = Triangulation(pts)
    edge_extension = np.multiply(pts[1], 2)

    _add_point_with_check(t, edge_extension)
    _check_triangulation_is_valid(t)

    simplex1 = tuple(range(dim + 1))
    simplex2 = tuple(range(1, dim + 2))

    assert t.simplices == {simplex1, simplex2}


@with_dimension
def test_adding_point_coplanar_with_a_face(dim):
    pts = _make_standard_simplex(dim)
    t = Triangulation(pts)
    face_extension = np.sum(pts[:-1], axis=0) * 2

    _add_point_with_check(t, face_extension)
    _check_triangulation_is_valid(t)

    simplex1 = tuple(range(dim + 1))
    simplex2 = tuple(range(1, dim + 2))

    assert t.simplices == {simplex1, simplex2}


@with_dimension
def test_adding_point_inside_circumscribed_circle(dim):
    pts = _make_standard_simplex(dim)
    t = Triangulation(pts)
    on_simplex = (0.6,) * dim

    _add_point_with_check(t, on_simplex)
    added_point = dim + 1  # *index* of added point

    _check_triangulation_is_valid(t)

    other_points = list(range(1, dim + 1))
    new_simplices = {
        (0, *points, added_point)
        for points in itertools.combinations(other_points, dim - 1)
    }
    assert new_simplices == t.simplices


@with_dimension
def test_triangulation_volume_is_less_than_bounding_box(dim):
    eps = 1e-8
    points = np.random.random((10, dim))  # all within the unit hypercube
    t = _make_triangulation(points)

    _check_triangulation_is_valid(t)
    assert np.sum(t.volumes()) < 1 + eps


@with_dimension
def test_triangulation_is_deterministic(dim):
    points = np.random.random((10, dim))
    t1 = _make_triangulation(points)
    t2 = _make_triangulation(points)
    assert t1.simplices == t2.simplices


@with_dimension
def test_initialisation_raises_when_not_enough_points(dim):
    deficient_simplex = _make_standard_simplex(dim)[:-1]

    with pytest.raises(ValueError):
        Triangulation(deficient_simplex)


@with_dimension
def test_initialisation_raises_when_points_coplanar(dim):
    zero_volume_simplex = _make_standard_simplex(dim)[:-1]

    new_point1 = np.average(zero_volume_simplex, axis=0)
    new_point2 = np.sum(zero_volume_simplex, axis=0)
    zero_volume_simplex = np.vstack((zero_volume_simplex, new_point1, new_point2))

    with pytest.raises(ValueError):
        Triangulation(zero_volume_simplex)


@with_dimension
def test_initialisation_accepts_more_than_one_simplex(dim):
    points = _make_standard_simplex(dim)
    new_point = [1.1] * dim  # Point oposing the origin but outside circumsphere
    points = np.vstack((points, new_point))

    tri = Triangulation(points)

    simplex1 = tuple(range(dim + 1))
    simplex2 = tuple(range(1, dim + 2))

    _check_triangulation_is_valid(tri)

    assert tri.simplices == {simplex1, simplex2}
