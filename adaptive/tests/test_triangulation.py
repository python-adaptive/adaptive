from collections import defaultdict, Counter
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


def _random_point_inside_standard_simplex(dim):
    coeffs = []
    for d in range(dim):
        coeffs.append((1 - sum(coeffs)) * np.random.random())
    coeffs = np.array(coeffs)
    # Sanity checks
    assert np.all(coeffs > 0) and not np.any(np.isclose(0, coeffs))
    assert np.sum(coeffs) < 1 and not np.isclose(1, np.sum(coeffs))
    return coeffs


def _check_simplices_are_valid(t):
    """Check that 'simplices' and 'vertex_to_simplices' are consistent."""
    vertex_to_simplices = defaultdict(set)
    for simplex in t.simplices:
        for vertex in simplex:
            vertex_to_simplices[vertex].add(simplex)
    assert vertex_to_simplices == t.vertex_to_simplices,\
           (t.vertex_to_simplices, vertex_to_simplices)


def _check_faces_are_valid(t):
    """Check that a 'dim-1'-D face is shared by no more than 2 simplices."""
    counts = Counter(t.faces())
    assert not any(i > 2 for i in counts.values()), counts


def _check_hull_is_valid(t):
    """Check that the stored hull is consistent with one computed from scratch."""
    counts = Counter(t.faces())
    hull = set(point
               for face, count in counts.items()
               if count == 1
               for point in face)

    assert t.hull == hull, (t.hull, hull)


def _check_triangulation_is_valid(t):
    _check_simplices_are_valid(t)
    _check_faces_are_valid(t)
    _check_hull_is_valid(t)


@with_dimension
def test_triangulation_of_standard_simplex_is_valid(dim):
    t = Triangulation(_make_standard_simplex(dim))
    expected_simplex = tuple(range(dim + 1))
    assert t.simplices == {expected_simplex}
    _check_triangulation_is_valid(t)
    assert np.isclose(t.volume(expected_simplex),
                      _standard_simplex_volume(dim))


@with_dimension
def test_zero_volume_initial_simplex_raises_exception(dim):
    points = np.random.random((dim - 1, dim))
    linearly_dependent_point = np.dot(np.random.random(dim - 1), points)
    points = np.vstack((np.zeros(dim), points, linearly_dependent_point))
    assert np.isclose(np.linalg.det(points[1:]), 0)  # sanity check

    with pytest.raises(ValueError):
        Triangulation(points)


@with_dimension
def test_adding_point_outside_standard_simplex_is_valid(dim):
    t = Triangulation(_make_standard_simplex(dim))
    t.add_point((1.1,) * dim)

    _check_triangulation_is_valid(t)
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
@pytest.mark.parametrize('provide_simplex', [True, False])
def test_adding_point_inside_standard_simplex_is_valid(dim, provide_simplex):
    t = Triangulation(_make_standard_simplex(dim))
    first_simplex = tuple(range(dim + 1))
    inside_simplex = _random_point_inside_standard_simplex(dim)
    if provide_simplex:
        t.add_point(inside_simplex, simplex=first_simplex)
    else:
        t.add_point(inside_simplex)
    added_point = dim + 1  # *index* of added point

    _check_triangulation_is_valid(t)
    assert len(t.simplices) == dim + 1
    assert all(added_point in simplex for simplex in t.simplices)

    volume = np.sum([t.volume(s) for s in t.simplices])
    assert np.isclose(volume, _standard_simplex_volume(dim))


@with_dimension
def test_adding_point_on_face_of_standard_simplex_is_valid(dim):
    if dim == 1:
        # there are no faces in 1D, so we'd end up adding an existing point
        return

    t = Triangulation(_make_standard_simplex(dim))
    centre_of_face = (1 / dim,) * dim
    t.add_point(centre_of_face)
    added_point = dim + 1  # *index* of added point

    _check_triangulation_is_valid(t)
    assert len(t.simplices) == dim
    assert all(added_point in simplex for simplex in t.simplices)

    volume = np.sum([t.volume(s) for s in t.simplices])
    assert np.isclose(volume, _standard_simplex_volume(dim))


@with_dimension
def test_triangulation_volume_is_less_than_bounding_box(dim):
    eps = 1e-8
    points = np.random.random((30, dim))  # all within the unit hypercube
    t = _make_triangulation(points)

    _check_triangulation_is_valid(t)
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert volume < 1+eps


@with_dimension
def test_triangulation_is_deterministic(dim):
    points = np.random.random((30, dim))
    t1 = _make_triangulation(points)
    t2 = _make_triangulation(points)
    assert t1.simplices == t2.simplices
