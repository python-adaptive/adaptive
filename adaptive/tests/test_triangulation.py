from ..learner.triangulation import Triangulation
import numpy as np


def test_initializing():
    c = [(0, 0), (1, 0), (0, 1)]
    t = Triangulation(c)
    assert t.simplices == {(0, 1, 2)}
    assert t.volume((0, 1, 2)) == 0.5


def test_adding_point_outside_simplex():
    c = [(0, 0), (1, 0), (0, 1)]
    t = Triangulation(c)
    t.add_point((1.1, 1.1))

    assert t.simplices == {(0, 1, 2), (1, 2, 3)}
    assert np.isclose(t.volume((0, 1, 2)), 0.5)
    assert t.volume((1, 2, 3)) > 0.5

    assert t.vertex_to_simplices[0] == {(0, 1, 2)}
    assert t.vertex_to_simplices[1] == {(0, 1, 2), (1, 2, 3)}
    assert t.vertex_to_simplices[2] == {(0, 1, 2), (1, 2, 3)}
    assert t.vertex_to_simplices[3] == {(1, 2, 3)}


def test_adding_point_inside_simplex():
    c = [(0, 0), (1, 0), (0, 1)]
    t = Triangulation(c)
    t.add_point((0.1, 0.1), simplex=(0, 1, 2))
    assert t.simplices == {(0, 1, 3), (0, 2, 3), (1, 2, 3)}
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert np.isclose(volume, 0.5)


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


def test_adding_many_points():
    dim = 2
    eps = 1e-8
    pts = np.random.random((100, dim))
    t = Triangulation(pts[:dim + 1])
    for p in pts[dim + 1:]:
        t.add_point(p)
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert volume < 1+eps


def test_3d_add_many():
    dim = 3
    eps = 1e-8
    pts = np.random.random((50, dim))
    t = Triangulation(pts[:dim + 1])
    for p in pts[dim + 1:]:
        t.add_point(p)
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert volume < 1 + eps


def test_4d_add_many():
    dim = 4
    eps = 1e-8
    pts = np.random.random((50, dim))
    t = Triangulation(pts[:dim + 1])
    for p in pts[dim + 1:]:
        t.add_point(p)
    volume = np.sum([t.volume(s) for s in t.simplices])
    assert volume < 1 + eps




