import numpy as np

import hypothesis.strategies as st
import pytest
from adaptive.tests.domain_utils import (
    a_few_points_inside,
    make_hypercube_domain,
    point_inside,
    point_on_shared_face,
    point_outside,
    points_inside,
    points_outside,
)
from hypothesis import given, settings


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
@settings(deadline=500)
def test_getting_points_are_unique(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    points = []
    for subdomain in domain.subdomains():
        p, _ = domain.insert_points(subdomain, 10)
        assert len(p) == len(set(p))
        points.extend(p)
    assert len(points) == len(set(points))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_sum_subvolumes_equals_volume(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    for x in xs:
        domain.split_at(x)
    for subdomain in domain.subdomains():
        assert np.isclose(domain.volume(subdomain), sum(domain.subvolumes(subdomain)))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_split_at_vertex_raises(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    x = data.draw(point_inside(domain))
    domain.split_at(x)
    with pytest.raises(ValueError):
        domain.split_at(x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_inserting_point_twice_raises(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    x = data.draw(point_inside(domain))
    domain.insert(x)
    with pytest.raises(ValueError):
        domain.insert(x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_insert_points_outside_domain_raises(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    x = data.draw(point_outside(domain))
    with pytest.raises(ValueError):
        domain.insert(x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_encloses(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))

    xin = data.draw(point_inside(domain))
    assert domain.encloses(xin)

    xout = data.draw(point_outside(domain))
    assert not domain.encloses(xout)

    xins = data.draw(points_inside(domain, 20))
    assert np.all(domain.encloses(xins))

    xouts = data.draw(points_outside(domain, 20))
    assert not np.any(domain.encloses(xouts))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_insert_point_outside_domain_raises(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    x = data.draw(point_outside(domain))
    with pytest.raises(ValueError):
        domain.insert(x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_split_at_point_outside_domain_raises(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    x = data.draw(point_outside(domain))
    with pytest.raises(ValueError):
        domain.split_at(x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_removing_domain_vertex_raises(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    x = data.draw(point_inside(domain))
    domain.split_at(x)
    with pytest.raises(ValueError):
        domain.remove(x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_removing_nonexistant_point_raises(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    x = data.draw(point_inside(domain))
    with pytest.raises(ValueError):
        domain.remove(x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_splitting_at_point_adds_to_vertices(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    for x in xs:
        domain.split_at(x)
    vertices = set(domain.vertices())
    assert all(x in vertices for x in xs)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_inserting_points_adds_to_subpoints(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    subdomains = dict()
    for x in xs:
        subdomains[x] = domain.insert(x)
    for x in xs:
        for subdomain in subdomains[x]:
            assert x in domain.subpoints(subdomain)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_inserting_then_removing_points_removes_from_subpoints(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    for x in xs:
        domain.insert(x)
    for x in xs:
        domain.remove(x)
    assert not any(domain.subpoints(s) for s in domain.subdomains())


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
@settings(deadline=500)
def test_inserting_then_splitting_at_points_removes_from_subpoints(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    for x in xs:
        domain.insert(x)
    for x in xs:
        domain.split_at(x)
    assert not any(domain.subpoints(s) for s in domain.subdomains())


@pytest.mark.parametrize("ndim", [1, 2, 3])
@given(data=st.data())
def test_clear_subdomains_removes_all_points(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    for x in xs:
        domain.insert(x)
    assert len(xs) == sum(len(domain.subpoints(s)) for s in domain.subdomains())
    domain.clear_subdomains()
    assert 0 == sum(len(domain.subpoints(s)) for s in domain.subdomains())


### Interval tests


### ConvexHull tests


@pytest.mark.parametrize("ndim", [2, 3])
@given(data=st.data())
def test_inserting_point_on_boundary_adds_to_all_subtriangulations(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    for x in xs:
        domain.split_at(x)
    x = data.draw(point_on_shared_face(domain, 1))
    affected_subdomains = domain.insert(x)
    assert all(x in set(domain.subpoints(s)) for s in affected_subdomains)


@pytest.mark.parametrize("ndim", [2, 3])
@given(data=st.data())
def test_split_at_reassigns_all_internal_points(data, ndim):
    domain = data.draw(make_hypercube_domain(ndim))
    xs = data.draw(a_few_points_inside(domain))

    for x in xs:
        domain.insert(x)
    _, new_subdomains = domain.split_at(xs[0])
    subpoints = set.union(*(set(domain.subpoints(s)) for s in new_subdomains))
    assert set(xs[1:]) == subpoints
