import itertools

import numpy as np

import hypothesis.strategies as st
from adaptive.learner.new_learnerND import ConvexHull, Interval

# This module contains utilities for producing domains and points inside and outside of them.
# Because we typically do not want to test very degenerate cases (e.g. points that are almost
# coincident, very large or very small) we prefer generating points in the interval [0, 1)
# using numpy.random, rather than drawing from Hypothesis' "floats" strategy.


# Return an iterator that yields matrices reflecting in the cartesian
# coordinate axes in 'ndim' dimensions.
def reflections(ndim):
    return map(np.diag, itertools.product([1, -1], repeat=ndim))


def point_inside_simplex(simplex):
    simplex = np.asarray(simplex)
    dim = simplex.shape[1]
    # Generate a point in the unit simplex.
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    # We avoid using Hypothesis to generate the points as it typically chooses
    # very annoying points, which we want to avoid testing for now.
    xb = np.random.rand(dim)
    xb = np.array(sorted(xb))
    xb[1:] = xb[1:] - xb[:-1]
    # Transform into the simplex we need
    v0, vecs = simplex[0], simplex[1:] - simplex[0]
    x = tuple(v0 + (vecs.T @ xb))
    return x


@st.composite
def points_inside(draw, domain, n):
    # Set the numpy random seed
    draw(st.random_module())
    if isinstance(domain, Interval):
        a, b = domain.bounds
        return a + (b - a) * np.random.rand(n)
    else:
        assert isinstance(domain, ConvexHull)
        tri = domain.triangulation
        simplices = list(tri.simplices)
        simplex = st.sampled_from(simplices).map(
            lambda simplex: [tri.vertices[s] for s in simplex]
        )
        # "point_inside_simplex" uses the numpy RNG, and we set the seed above.
        # Together this means we're almost guaranteed not to get coinciding points.
        # Note that we draw from the 'simplex' strategy on each iteration, so we
        # distribute the points between the different simplices in the domain.
        return [tuple(point_inside_simplex(draw(simplex))) for _ in range(n)]


@st.composite
def point_inside(draw, domain):
    return draw(points_inside(domain, 1))[0]


@st.composite
def a_few_points_inside(draw, domain):
    n = draw(st.integers(3, 20))
    return draw(points_inside(domain, n))


@st.composite
def points_outside(draw, domain, n):
    # set numpy random seed
    draw(st.random_module())

    if isinstance(domain, Interval):
        a, b = domain.bounds
        ndim = 1
    else:
        assert isinstance(domain, ConvexHull)
        hull = domain.bounds
        points = hull.points[hull.vertices]
        ndim = points.shape[1]
        a, b = points.min(axis=0)[None, :], points.max(axis=0)[None, :]

    # Generate a point outside the bounding box of the domain.
    center = (a + b) / 2
    border = (b - a) / 2
    r = border + 10 * border * np.random.rand(n, ndim)
    quadrant = np.sign(np.random.rand(n, ndim) - 0.5)
    assert not np.any(quadrant == 0)
    return center + quadrant * r


@st.composite
def point_outside(draw, domain):
    return draw(points_outside(domain, 1))[0]


@st.composite
def point_on_shared_face(draw, domain, dim):
    # Return a point that is shared by at least 2 subdomains
    assert isinstance(domain, ConvexHull)
    assert 0 < dim < domain.ndim

    # Set the numpy random seed
    draw(st.random_module())

    tri = domain.triangulation

    for face in tri.faces(dim + 1):
        containing_subdomains = tri.containing(face)
        if len(containing_subdomains) > 1:
            break

    vertices = np.array([tri.vertices[i] for i in face])

    xb = np.random.rand(dim)

    x = tuple(vertices[0] + xb @ (vertices[1:] - vertices[0]))

    assert all(tri.point_in_simplex(x, s) for s in containing_subdomains)

    return x


@st.composite
def make_random_domain(draw, ndim, fill=True):
    # Set the numpy random seed
    draw(st.random_module())

    if ndim == 1:
        a, b = sorted(np.random.rand(2) - 0.5)
        domain = Interval(a, b)
    else:
        # Generate points in a hypercube around the origin
        points = np.random.rand(10, ndim) - 0.5
        domain = ConvexHull(points)
    return domain


@st.composite
def make_hypercube_domain(draw, ndim, fill=True):
    # Set the numpy random seed
    draw(st.random_module())
    limit = np.random.rand()

    if ndim == 1:
        subdomain = Interval(-limit, limit)
    else:
        point = np.full(ndim, limit)
        boundary_points = [r @ point for r in reflections(ndim)]
        subdomain = ConvexHull(boundary_points)
    return subdomain
