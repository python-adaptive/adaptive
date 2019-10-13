import abc
import functools
import itertools
from collections import defaultdict

import numpy as np
import scipy.linalg
import scipy.spatial
from sortedcontainers import SortedList

from adaptive.learner.triangulation import Triangulation, circumsphere, point_in_simplex

__all__ = ["Domain", "Interval", "ConvexHull"]


class Domain(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def insert_points(self, subdomain, n):
        """Insert 'n' points into 'subdomain'.

        Returns
        -------
        affected_subdomains : Iterable of subdomains
            If some points were added on the boundary of 'subdomain'
            then they will also have been added to the neighboring
            subdomains.
        """

    @abc.abstractmethod
    def insert(self, x):
        """Insert 'x' into any subdomains to which it belongs.

        Returns
        -------
        affected_subdomains : Iterable of subdomains
            The subdomains to which 'x' was added.

        Raises
        ------
        ValueError : if x is outside the domain or exists already
        """

    @abc.abstractmethod
    def remove(self, x):
        """Remove 'x' from any subdomains to which it belongs.

        Returns
        -------
        affected_subdomains : Iterable of subdomains
            The subdomains from which 'x' was removed.

        Raises
        ------
        ValueError : if x is a subdomain vertex
        ValueError : if x is not in any subdomain
        """

    @abc.abstractmethod
    def split_at(self, x):
        """Split the domain at 'x'.

        Removes and adds subdomains.

        Returns
        -------
        old_subdomains : list of subdomains
            The subdomains that were removed when splitting at 'x'.
        new_subdomains : list of subdomains
            The subdomains that were added when splitting at 'x'.

        Raises
        ------
        ValueError : if x is outside of the domain or exists already
        """

    @abc.abstractmethod
    def which_subdomains(self, x):
        """Return the subdomains that contains 'x'.

        Return
        ------
        subdomains : Iterable of subdomains
            The subdomains to which 'x' belongs.

        Raises
        ------
        ValueError : if x is outside of the domain
        """

    @abc.abstractmethod
    def encloses(self, points):
        """Return whether the domain encloses the points

        Parameters
        ----------
        points : a point or sequence of points

        Returns
        -------
        Boolean (if a single point was provided) or an array of booleans
        if a sequence of points was provided) that is True when
        the domain encloses the point.
        """

    @abc.abstractmethod
    def vertices(self):
        """Returns the vertices of the domain."""

    @abc.abstractmethod
    def neighbors(self, subdomain, n=1):
        "Return all neighboring subdomains up to degree 'n'."

    @abc.abstractmethod
    def subdomains(self):
        "Return all the subdomains in the domain."

    @abc.abstractmethod
    def subpoints(self, subdomain):
        "Return all points in the interior of a subdomain."

    @abc.abstractmethod
    def clear_subdomains(self):
        """Remove all points from the interior of subdomains.

        Returns
        -------
        subdomains : the subdomains who's interior points were removed
        """

    @abc.abstractmethod
    def volume(self, subdomain):
        "Return the volume of a subdomain."

    @abc.abstractmethod
    def subvolumes(self, subdomain):
        "Return the volumes of the sub-subdomains."


def _choose_point_in_subinterval(a, b):
    m = a + (b - a) / 2
    if not a < m < b:
        raise ValueError("{} cannot be split further".format((a, b)))
    return m


class Interval(Domain):
    """A 1D domain (an interval).

    Subdomains are pairs of floats (a, b).
    """

    def __init__(self, a, b):
        if a >= b:
            raise ValueError("'a' must be less than 'b'")

        # If a sub-interval contains points in its interior, they are stored
        # in 'sub_intervals' in a SortedList.
        self.bounds = (a, b)
        self.sub_intervals = dict()
        self.points = SortedList([a, b])
        self.ndim = 1

    def insert_points(self, subdomain, n, *, _check_membership=True):
        if n <= 0:
            raise ValueError("n must be positive")
        if _check_membership and not self.contains_subdomain(subdomain):
            raise ValueError("{} is not present in this interval".format(subdomain))
        try:
            p = self.sub_intervals[subdomain]
        except KeyError:  # No points yet in the interior of this subdomain
            p = SortedList(subdomain)
            self.sub_intervals[subdomain] = p

        # Choose new points in the centre of the largest subdomain
        # of this subinterval.
        points = []
        subsubdomains = SortedList(zip(p, p.islice(1)), key=self.volume)
        for _ in range(n):
            a, b = subsubdomains.pop()
            m = _choose_point_in_subinterval(a, b)
            subsubdomains.update([(a, m), (m, b)])
            points.append(m)
        p.update(points)

        return points, [subdomain]

    def insert(self, x, *, _check_membership=True):
        if _check_membership:
            a, b = self.bounds
            if not (a <= x <= b):
                raise ValueError("{} is outside of this interval".format(x))

        p = self.points
        i = p.bisect_left(x)
        if p[i] == x:
            raise ValueError("{} exists in this interval already".format(x))
        subdomain = (a, b) = p[i - 1], p[i]

        try:
            p = self.sub_intervals[subdomain]
        except KeyError:
            self.sub_intervals[subdomain] = SortedList([a, x, b])
        else:
            if x in p:
                raise ValueError("{} exists in a subinterval already".format(x))
            p.add(x)

        return [subdomain]

    def remove(self, x, *, _check_membership=True):
        if _check_membership:
            a, b = self.bounds
            if not (a <= x <= b):
                raise ValueError("{} is outside of this interval".format(x))

        p = self.points
        i = p.bisect_left(x)
        if p[i] == x:
            raise ValueError("Cannot remove subdomain vertices")
        subdomain = (p[i - 1], p[i])

        try:
            sub_points = self.sub_intervals[subdomain]
        except KeyError:
            raise ValueError("{} not in any subdomain".format(x))
        else:
            sub_points.remove(x)
            if len(sub_points) == 2:
                del self.sub_intervals[subdomain]
            return [subdomain]

    def split_at(self, x, *, _check_membership=True):
        a, b = self.bounds
        if _check_membership:
            if not (a <= x <= b):
                raise ValueError("Can only split at points within the interval")

        p = self.points
        i = p.bisect_left(x)
        if p[i] == x:
            raise ValueError("Cannot split at an existing point")
        a, b = old_interval = p[i - 1], p[i]
        new_intervals = [(a, x), (x, b)]

        p.add(x)
        try:
            sub_points = self.sub_intervals.pop(old_interval)
        except KeyError:
            pass
        else:
            # Update subintervals
            for ival in new_intervals:
                new_sub_points = SortedList(sub_points.irange(*ival))
                if x not in new_sub_points:
                    # This should add 'x' to the start or the end
                    new_sub_points.add(x)
                if len(new_sub_points) > 2:
                    # We don't store subintervals if they don't contain
                    # any points in their interior.
                    self.sub_intervals[ival] = new_sub_points

        return [old_interval], new_intervals

    def which_subdomains(self, x):
        a, b = self.bounds
        if not (a <= x <= b):
            raise ValueError("{} is outside the interval".format(x))
        p = self.points
        i = p.bisect_left(x)
        if p[i] != x:
            # general point inside a subinterval
            return [(p[i - 1], p[i])]
        else:
            # boundary of a subinterval
            neighbors = []
            if i > 0:
                neighbors.append((p[i - 1], p[i]))
            if i < len(p) - 1:
                neighbors.append((p[i], p[i + 1]))
            return neighbors

    def contains_subdomain(self, subdomain):
        a, b = subdomain
        try:
            ia = self.points.index(a)
            ib = self.points.index(b)
        except ValueError:
            return False
        return ia + 1 == ib

    def vertices(self):
        return self.points

    def encloses(self, points):
        a, b = self.bounds
        points = np.asarray(points)
        if points.shape == ():  # single point
            return a <= points <= b
        else:
            return np.logical_and(a <= points, points <= b)

    def neighbors(self, subdomain, n=1):
        a, b = subdomain
        p = self.points
        ia = p.index(a)
        neighbors = []
        for i in range(n):
            if ia - i > 0:  # left neighbor exists
                neighbors.append((p[ia - i - 1], p[ia - i]))
            if ia + i < len(p) - 2:  # right neighbor exists
                neighbors.append((p[ia + i + 1], p[ia + i + 2]))
        return neighbors

    def subdomains(self):
        p = self.points
        return zip(p, p.islice(1))

    def subpoints(self, subdomain, *, _check_membership=True):
        if _check_membership and not self.contains_subdomain(subdomain):
            raise ValueError("{} is not present in this interval".format(subdomain))
        try:
            p = self.sub_intervals[subdomain]
        except KeyError:
            return []
        else:
            # subinterval points contain the vertex points
            return p[1:-1]

    def clear_subdomains(self):
        subdomains = list(self.sub_intervals.keys())
        self.sub_intervals = dict()
        return subdomains

    def volume(self, subdomain):
        a, b = subdomain
        return b - a

    def subvolumes(self, subdomain):
        try:
            p = self.sub_intervals[subdomain]
        except KeyError:
            return [self.volume(subdomain)]
        else:
            return [self.volume(s) for s in zip(p, p.islice(1))]


def _choose_point_in_simplex(simplex, transform=None):
    """Choose a good point at which to split a simplex.

    Parameters
    ----------
    simplex : (n+1, n) array
        The simplex vertices
    transform : (n, n) array
        The linear transform to apply to the simplex vertices
        before determining which point to choose. Must be
        invertible.

    Returns
    -------
    point : (n,) array
        The point that was chosen in the simplex
    """
    if transform is not None:
        simplex = np.dot(simplex, transform)

    # Choose center only if the shape of the simplex is nice,
    # otherwise: the center the longest edge
    center, _radius = circumsphere(simplex)
    if point_in_simplex(center, simplex):
        point = np.average(simplex, axis=0)
    else:
        distances = scipy.spatial.distance.pdist(simplex)
        distance_matrix = scipy.spatial.distance.squareform(distances)
        i, j = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        point = (simplex[i, :] + simplex[j, :]) / 2

    if transform is not None:
        point = np.linalg.solve(transform, point)  # undo the transform

    return tuple(point)


def _simplex_facets(ndim):
    """Return the facets of a simplex in 'ndim' dimensions

    Parameters
    ----------
    ndim : positive int

    Returns
    -------
    facets : Iterable of integer tuples
        Contains 'ndim + 1' tuples, and each tuple contains
        'ndim' integers.

    Examples
    --------
    In 2D a simplex is a triangle (3 points) and the facets are the lines
    joining these points:

    >>> list(_simplex_facets(2))
    [(0, 1), (0, 2), (1, 2)]
    """
    return itertools.combinations(range(ndim + 1), ndim)


def _boundary_equations(simplex):
    """Return the set of equations defining the boundary of a simplex

    Parameters
    ----------
    simplex : (N + 1, N) float array-like
        The vertices of an N-dimensional simplex.

    Returns
    -------
    A : (N + 1, N) float array
        Each row is a normal vector to a facet of 'simplex'.
        The facets are in the same order as returned by
        '_simplex_facets(N)'.
    b : (N + 1,) float array
        Each element is the offset from the origin of the
        corresponding facet of 'simplex'

    Notes
    -----

    This is slower than using scipy.spatial.ConvexHull, however the ordering
    of the equations as returned by scipy.spatial.ConvexHull is not clear.

    Care is not taken to orient the facets to point out of the simplex; the
    equations should only be used for verifying if a point lies on a boundary,
    rather than if it lies inside the simplex.

    Examples
    --------
    >>> simplex = [(0, 0), (1, 0), (0, 1)]
    >>> A, b =  _boundary_equations(simplex)
    >>> x = [0.5, 0]
    >>> which_boundary = np.isclose(A @ x + b, 0)
    >>> # facet #0 is the line between (0, 0) and (1, 0)
    >>> assert which_boundary[0] == True
    """
    points = np.asarray(simplex)
    ndim = points.shape[1]
    assert points.shape == (ndim + 1, ndim)
    A = np.empty((ndim + 1, ndim), dtype=float)
    b = np.empty((ndim + 1), dtype=float)
    for i, (x0, *v) in enumerate(_simplex_facets(ndim)):
        facet_tangent_space = points[list(v)] - points[x0]
        facet_normal = scipy.linalg.null_space(facet_tangent_space).squeeze()
        A[i, :] = facet_normal
        b[i] = -np.dot(points[x0], facet_normal)
    return A, b


def _on_which_boundary(equations, x, eps=1e-8):
    """Returns the simplex boundary on which 'x' is found.

    Parameters
    ----------
    equations : the output of _boundary_equations
        The equations defining a simplex in 'N' dimensions
    x : (N,) float array-like

    Returns
    -------
    None if 'x' is not on a simplex boundary.
    Otherwise, returns a tuple containing integers defining
    the boundary on which 'x' is found.

    Examples
    --------
    >>> simplex = [(0., 0.), (2., 0.), (0., 4.)]
    >>> eq = _boundary_equations(simplex)
    >>> x = [0.5, 0.]
    >>> _on_which_boundary(eq, x) == (0, 1)
    >>> assert boundary == (0, 1)
    >>> x = [2., 0.]
    >>> _on_which_boundary(eq, x) == (1,)
    """
    ndim = len(x)
    A, b = equations
    assert len(b) == ndim + 1
    on_boundary = np.isclose(A @ x + b, 0, atol=1e-8)
    if not any(on_boundary):
        return None
    # The point is on the boundary of all the following facets
    facets = [facet for i, facet in enumerate(_simplex_facets(ndim)) if on_boundary[i]]
    # If the point is on the boundary of more than 1 facet, then it is on a lower-dimension facet.
    boundary_facet = set.intersection(*map(set, facets))
    return tuple(sorted(boundary_facet))


def _make_new_subtriangulation(points):
    points = np.asarray(points)
    ndim = points.shape[1]
    boundary_points = points[: ndim + 1]
    subtri = Triangulation(points)
    subtri.on_which_boundary = functools.partial(
        _on_which_boundary, _boundary_equations(boundary_points)
    )
    return subtri


class ConvexHull(Domain):
    """A convex hull domain in $ℝ^N$ (N >=2).

    Subdomains are simplices represented by integer tuples of length (N + 1).
    """

    def __init__(self, points):
        hull = scipy.spatial.ConvexHull(points)

        self.bounds = hull
        self.triangulation = Triangulation(hull.points[hull.vertices])
        # if a subdomain has interior points, then it appears as a key
        # in 'sub_triangulations' and maps to a 'Triangulation' of the
        # interior of the subdomain. By definition the triangulation
        # is over a simplex, and the first 'ndim + 1' points in the
        # triangulation are the boundary points.
        self.sub_triangulations = dict()
        self.ndim = self.bounds.points.shape[1]

        # As an optimization we store any points inserted with 'insert_points'
        # and 'insert' and point to the subdomains to which they belong. This
        # allows 'which_subdomains' and 'split_at' to work faster when given points
        # that were previously added with 'insert' or 'insert_points'
        self.subpoints_to_subdomains = defaultdict(set)

    @property
    def bounding_box(self):
        hull_points = self.bounds.points[self.bounds.vertices]
        return tuple(zip(hull_points.min(axis=0), hull_points.max(axis=0)))

    def _get_subtriangulation(self, subdomain):
        try:
            subtri = self.sub_triangulations[subdomain]
        except KeyError:  # No points in the interior of this subdomain yet
            points = [self.triangulation.vertices[x] for x in subdomain]
            subtri = _make_new_subtriangulation(points)
            self.sub_triangulations[subdomain] = subtri
        return subtri

    def insert_points(self, subdomain, n, *, _check_membership=True):
        if n <= 0:
            raise ValueError("n must be positive")
        tri = self.triangulation
        if _check_membership and subdomain not in tri.simplices:
            raise ValueError("{} is not present in this domain".format(subdomain))

        subtri = self._get_subtriangulation(subdomain)

        # Choose the largest volume sub-simplex and insert a point into it.
        # Also insert the point into neighboring subdomains if it was chosen
        # on the subdomain boundary.
        points = []
        affected_subdomains = {subdomain}
        for _ in range(n):
            # O(N) in the number of sub-simplices, but typically we only have a few
            largest_simplex = max(subtri.simplices, key=subtri.volume)
            simplex_vertices = np.array([subtri.vertices[s] for s in largest_simplex])
            point = _choose_point_in_simplex(simplex_vertices)
            points.append(point)
            subtri.add_point(point, largest_simplex)
            self.subpoints_to_subdomains[point].add(subdomain)
            # If the point was added to a boundary of the subdomain we should
            # add it to the neighboring subdomains.
            boundary = subtri.on_which_boundary(point)
            if boundary is not None:
                # Convert subtriangulation indices to triangulation indices
                boundary = tuple(sorted(subdomain[i] for i in boundary))
                neighbors = set(tri.containing(boundary))
                neighbors.remove(subdomain)
                for sd in neighbors:
                    self._get_subtriangulation(sd).add_point(point)
                affected_subdomains.update(neighbors)
                self.subpoints_to_subdomains[point].update(neighbors)

        return [tuple(p) for p in points], affected_subdomains

    def insert(self, x, *, _check_membership=True):
        x = tuple(x)
        # XXX: O(N) in the number of simplices
        affected_subdomains = self.which_subdomains(x)
        if not affected_subdomains:
            raise ValueError("{} is not present in this domain".format(x))
        for subdomain in affected_subdomains:
            subtri = self._get_subtriangulation(subdomain)
            if x in subtri.vertices:  # O(N) in the number of vertices
                raise ValueError("{} exists in a subinterval already".format(x))
            subtri.add_point(x)
        self.subpoints_to_subdomains[x].update(affected_subdomains)

        return affected_subdomains

    def remove(self, x):
        x = tuple(x)
        try:
            affected_subdomains = self.subpoints_to_subdomains.pop(x)
        except KeyError:
            raise ValueError("Can only remove points inside subdomains")
        for subdomain in affected_subdomains:
            # Check that it's not a vertex of the subdomain
            subtri = self.sub_triangulations[subdomain]
            assert x in subtri.vertices
            points = [v for v in subtri.vertices if v != x]
            if len(points) == self.ndim + 1:
                # No more points inside the subdomain
                del self.sub_triangulations[subdomain]
            else:
                # Rebuild the subtriangulation from scratch
                self.sub_triangulations[subdomain] = _make_new_subtriangulation(points)

        return affected_subdomains

    def split_at(self, x, *, _check_membership=True):
        x = tuple(x)
        tri = self.triangulation
        try:
            containing_subdomains = self.subpoints_to_subdomains.pop(x)
            # Only need a single subdomaing 'x' to make 'tri.add_point' fast.
            subdomain = next(iter(containing_subdomains))
        except KeyError:
            # XXX: O(N) in the number of simplices.
            subdomain = tri.locate_point(x)
            if not subdomain:
                raise ValueError("Can only split at points within the domain.")

        old_subdomains, new_subdomains = tri.add_point(x, subdomain)

        if _check_membership:
            assert not any(s in self.sub_triangulations for s in new_subdomains)

        # Re-assign all the interior points of 'old_subdomains' to 'new_subdomains'

        # Keep the interior points as a set, because interior points on a shared face
        # appear in the subtriangulations of both the neighboring simplices, and we
        # don't want those points to appear twice.
        interior_points = set()
        for d in old_subdomains:
            try:
                subtri = self.sub_triangulations.pop(d)
            except KeyError:
                continue
            else:
                # Get all points in the subtriangulation except the boundary
                # points. Because a subtriangulation is always defined over
                # a simplex, the first ndim + 1 points are the boundary points.
                interior = [v for v in subtri.vertices[self.ndim + 1 :] if v != x]
                for v in interior:
                    s = self.subpoints_to_subdomains[v]
                    s.remove(d)
                    if not s:
                        del self.subpoints_to_subdomains[v]
                interior_points.update(interior)
        for p in interior_points:
            # Try to add 'p' to all the new subdomains. It may belong to more than 1
            # if it lies on a subdomain boundary.
            p_was_added = False
            for subdomain in new_subdomains:
                if tri.point_in_simplex(p, subdomain):
                    subtri = self._get_subtriangulation(subdomain)
                    subtri.add_point(p)
                    self.subpoints_to_subdomains[p].add(subdomain)
                    p_was_added = True
            assert (
                p_was_added
            ), "{} was not in the interior of any new simplices".format(x)

        return old_subdomains, new_subdomains

    def which_subdomains(self, x):
        x = tuple(x)
        tri = self.triangulation
        if x in self.subpoints_to_subdomains:
            subdomains = self.subpoints_to_subdomains[x]
        else:
            # XXX: O(N) in the number of simplices
            subdomains = [s for s in tri.simplices if tri.point_in_simplex(x, s)]
            if not subdomains:
                raise ValueError("{} is not in the domain".format(x))
        return list(subdomains)

    def contains_subdomain(self, subdomain):
        return subdomain in self.triangulation.simplices

    def transform(self, x):
        # XXX: implement this
        raise NotImplementedError()

    def neighbors(self, subdomain, n=1):
        tri = self.triangulation
        neighbors = {subdomain}
        for _ in range(n):
            for face in list(tri.faces(simplices=neighbors)):
                neighbors.update(tri.containing(face))
        neighbors.remove(subdomain)
        return neighbors

    def subdomains(self):
        return self.triangulation.simplices

    def encloses(self, points):
        points = np.asarray(points).T
        A, b = self.bounds.equations[:, :-1], self.bounds.equations[:, -1:]
        if len(points.shape) == 1:
            points = points[:, None]
        return np.all(A @ points + b <= 0, axis=0)

    def vertices(self):
        return self.triangulation.vertices

    def subpoints(self, subdomain, *, _check_membership=True):
        if _check_membership and not self.contains_subdomain(subdomain):
            raise ValueError("{} is not present in this domain".format(subdomain))
        try:
            subtri = self.sub_triangulations[subdomain]
        except KeyError:
            return []
        else:
            # Subtriangulations are, by definition, over simplices. This means
            # that the first ndim + 1 points are the simplex vertices, which we skip
            return subtri.vertices[self.ndim + 1 :]

    def clear_subdomains(self):
        sub_triangulations = list(self.sub_triangulations.keys())
        self.sub_triangulations = dict()
        return sub_triangulations

    def volume(self, subdomain):
        return self.triangulation.volume(subdomain)

    def subvolumes(self, subdomain):
        try:
            subtri = self.sub_triangulations[subdomain]
        except KeyError:
            return [self.triangulation.volume(subdomain)]
        else:
            return [subtri.volume(s) for s in subtri.simplices]
