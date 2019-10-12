import itertools

import numpy as np
import scipy.linalg
import scipy.spatial

import hypothesis.strategies as st
from adaptive.learner.new_learnerND import ConvexHull, Interval
from hypothesis.extra import numpy as hynp


def reflections(ndim):
    return map(np.diag, itertools.product([1, -1], repeat=ndim))


reals = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
positive_reals = st.floats(
    min_value=1e-3, max_value=100, allow_nan=False, allow_infinity=False
)


@st.composite
def point(draw, ndim):
    return draw(reals if ndim == 1 else st.tuples(*[reals] * ndim))


def unique_vectors(xs):
    xs = np.asarray(xs)
    if len(xs.shape) == 1:
        xs = xs[:, None]
    c = np.max(np.linalg.norm(xs, axis=1))
    if c == 0:
        return False
    d = scipy.spatial.distance_matrix(xs, xs)
    d = np.extract(1 - np.identity(d.shape[0]), d)
    return not np.any(d < 1e-3 / c)


@st.composite
def point_inside_simplex(draw, simplex):
    simplex = np.asarray(simplex)
    dim = simplex.shape[1]
    # Set the numpy random seed
    draw(st.random_module())
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
    kwargs = dict(
        allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True
    )
    if isinstance(domain, Interval):
        a, b = domain.bounds
        eps = (b - a) * 1e-2
        x = st.floats(min_value=(a + eps), max_value=(b - eps), **kwargs)
    else:
        assert isinstance(domain, ConvexHull)
        tri = domain.triangulation
        simplices = list(tri.simplices)
        simplex = draw(st.sampled_from(simplices))
        vertices = [tri.vertices[s] for s in simplex]
        x = point_inside_simplex(vertices)

    xs = st.tuples(*[x] * n).filter(unique_vectors)
    return draw(xs)


@st.composite
def point_inside(draw, domain):
    return draw(points_inside(domain, 1))[0]


@st.composite
def a_few_points_inside(draw, domain):
    n = draw(st.integers(3, 20))
    return draw(points_inside(domain, n))


@st.composite
def point_outside(draw, domain):
    kwargs = dict(allow_nan=False, allow_infinity=False)
    if isinstance(domain, Interval):
        a, b = domain.bounds
        length = b - a
        before_domain = st.floats(a - 10 * length, a, **kwargs)
        after_domain = st.floats(b, b + 10 * length, **kwargs)
        x = before_domain | after_domain
    else:
        assert isinstance(domain, ConvexHull)
        hull = domain.bounds
        # Generate point between bounding box and bounding box * 10
        points = hull.points[hull.vertices]
        x = st.tuples(
            *[
                (
                    st.floats(min_value=a - 10 * (b - a), max_value=a, **kwargs)
                    | st.floats(min_value=b, max_value=b + 10 * (b - a), **kwargs)
                )
                for a, b in zip(points.min(axis=0), points.max(axis=0))
            ]
        )

    return draw(x)


@st.composite
def point_on_shared_face(draw, domain, dim):
    # Return a point that is shared by at least 2 subdomains
    assert isinstance(domain, ConvexHull)
    assert 0 < dim < domain.ndim

    tri = domain.triangulation

    for face in tri.faces(dim + 1):
        containing_subdomains = tri.containing(face)
        if len(containing_subdomains) > 1:
            break

    vertices = np.array([tri.vertices[i] for i in face])

    f = st.floats(1e-3, 1 - 1e-3, allow_nan=False, allow_infinity=False)
    xb = draw(st.tuples(*[f] * dim))

    x = tuple(vertices[0] + xb @ (vertices[1:] - vertices[0]))

    assert all(tri.point_in_simplex(x, s) for s in containing_subdomains)

    return x


@st.composite
def make_random_domain(draw, ndim, fill=True):
    if ndim == 1:
        limits = draw(st.tuples(reals, reals).map(sorted).filter(lambda x: x[0] < x[1]))
        domain = Interval(*limits)
    else:
        points = draw(
            hynp.arrays(np.float, (10, ndim), elements=reals, unique=True).filter(
                unique_vectors
            )
        )
        domain = ConvexHull(points)
    return domain


@st.composite
def make_hypercube_domain(draw, ndim, fill=True):
    if ndim == 1:
        limit = draw(positive_reals)
        subdomain = Interval(-limit, limit)
    else:
        x = draw(positive_reals)
        point = np.full(ndim, x)
        boundary_points = [r @ point for r in reflections(ndim)]
        subdomain = ConvexHull(boundary_points)
    return subdomain
