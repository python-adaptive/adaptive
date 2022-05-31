from collections import Counter
from collections.abc import Iterable, Sized
from itertools import chain, combinations
from math import factorial, sqrt

import scipy.spatial
from numpy import abs as np_abs
from numpy import (
    array,
    asarray,
    average,
    concatenate,
    dot,
    eye,
    mean,
    ones,
    square,
    subtract,
)
from numpy import sum as np_sum
from numpy import zeros
from numpy.linalg import det as ndet
from numpy.linalg import matrix_rank, norm, slogdet, solve


def fast_norm(v):
    """Take the vector norm for len 2, 3 vectors.
    Defaults to a square root of the dot product for larger vectors.

    Note that for large vectors, it is possible for integer overflow to occur.
    For instance:
    vec = [49024, 59454, 12599, -63721, 18517, 27961]
    dot(vec, vec) = -1602973744

    """
    len_v = len(v)
    if len_v == 2:
        return sqrt(v[0] * v[0] + v[1] * v[1])
    if len_v == 3:
        return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return sqrt(dot(v, v))


def fast_2d_point_in_simplex(point, simplex, eps=1e-8):
    (p0x, p0y), (p1x, p1y), (p2x, p2y) = simplex
    px, py = point

    area = 0.5 * (-p1y * p2x + p0y * (p2x - p1x) + p1x * p2y + p0x * (p1y - p2y))

    s = 1 / (2 * area) * (+p0y * p2x + (p2y - p0y) * px - p0x * p2y + (p0x - p2x) * py)
    if s < -eps or s > 1 + eps:
        return False
    t = 1 / (2 * area) * (+p0x * p1y + (p0y - p1y) * px - p0y * p1x + (p1x - p0x) * py)

    return (t >= -eps) and (s + t <= 1 + eps)


def point_in_simplex(point, simplex, eps=1e-8):
    if len(point) == 2:
        return fast_2d_point_in_simplex(point, simplex, eps)

    x0 = array(simplex[0], dtype=float)
    vectors = array(simplex[1:], dtype=float) - x0
    alpha = solve(vectors.T, point - x0)

    return all(alpha > -eps) and sum(alpha) < 1 + eps


def fast_2d_circumcircle(points):
    """Compute the center and radius of the circumscribed circle of a triangle

    Parameters
    ----------
    points: 2D array-like
        the points of the triangle to investigate

    Returns
    -------
    tuple
        (center point : tuple(float), radius: float)
    """
    points = array(points)
    # transform to relative coordinates
    pts = points[1:] - points[0]

    (x1, y1), (x2, y2) = pts
    # compute the length squared
    l1 = x1 * x1 + y1 * y1
    l2 = x2 * x2 + y2 * y2

    # compute some determinants
    dx = +l1 * y2 - l2 * y1
    dy = -l1 * x2 + l2 * x1
    aa = +x1 * y2 - x2 * y1
    a = 2 * aa

    # compute center
    x = dx / a
    y = dy / a
    radius = sqrt(x * x + y * y)  # radius = norm([x, y])

    return (x + points[0][0], y + points[0][1]), radius


def fast_3d_circumcircle(points):
    """Compute the center and radius of the circumscribed sphere of a simplex.

    Parameters
    ----------
    points: 2D array-like
        the points of the triangle to investigate

    Returns
    -------
    tuple
        (center point : tuple(float), radius: float)
    """
    points = array(points)
    pts = points[1:] - points[0]

    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = pts

    l1 = x1 * x1 + y1 * y1 + z1 * z1
    l2 = x2 * x2 + y2 * y2 + z2 * z2
    l3 = x3 * x3 + y3 * y3 + z3 * z3

    # Compute some determinants:
    dx = +l1 * (y2 * z3 - z2 * y3) - l2 * (y1 * z3 - z1 * y3) + l3 * (y1 * z2 - z1 * y2)
    dy = +l1 * (x2 * z3 - z2 * x3) - l2 * (x1 * z3 - z1 * x3) + l3 * (x1 * z2 - z1 * x2)
    dz = +l1 * (x2 * y3 - y2 * x3) - l2 * (x1 * y3 - y1 * x3) + l3 * (x1 * y2 - y1 * x2)
    aa = +x1 * (y2 * z3 - z2 * y3) - x2 * (y1 * z3 - z1 * y3) + x3 * (y1 * z2 - z1 * y2)
    a = 2 * aa

    center = (dx / a, -dy / a, dz / a)
    radius = fast_norm(center)
    center = (
        center[0] + points[0][0],
        center[1] + points[0][1],
        center[2] + points[0][2],
    )

    return center, radius


def fast_det(matrix):
    matrix = asarray(matrix, dtype=float)
    if matrix.shape == (2, 2):
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    elif matrix.shape == (3, 3):
        a, b, c, d, e, f, g, h, i = matrix.ravel()
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    else:
        return ndet(matrix)


def circumsphere(pts):
    """Compute the center and radius of a N dimension sphere which touches each point in pts.

    Parameters
    ----------
    pts : array-like, of shape (N-dim + 1, N-dim)
        The points for which we would like to compute a circumsphere.

    Returns
    -------
    center : tuple of floats of size N-dim
    radius : a positive float
    A valid center and radius, if a circumsphere is possible, and no points are repeated.
    If points are repeated, or a circumsphere is not possible, will return nans, and a
    ZeroDivisionError may occur.
    Will fail for matrices which are not (N-dim + 1, N-dim) in size due to non-square determinants:
    will raise numpy.linalg.LinAlgError.
    May fail for points that are integers (due to 32bit integer overflow).
    """

    dim = len(pts) - 1
    if dim == 2:
        return fast_2d_circumcircle(pts)
    if dim == 3:
        return fast_3d_circumcircle(pts)

    # Modified method from http://mathworld.wolfram.com/Circumsphere.html
    mat = array([[np_sum(square(pt)), *pt, 1] for pt in pts])
    center = zeros(dim)
    a = 1 / (2 * ndet(mat[:, 1:]))
    factor = a
    # Use ind to index into the matrix columns
    ind = ones((dim + 2,), bool)
    for i in range(1, len(pts)):
        ind[i - 1] = True
        ind[i] = False
        center[i - 1] = factor * ndet(mat[:, ind])
        factor *= -1

    # Use subtract as we don't know the type of x0.
    x0 = pts[0]
    vec = subtract(center, x0)
    # Vector norm.
    radius = sqrt(dot(vec, vec))

    return tuple(center), radius


def orientation(face, origin):
    """Compute the orientation of the face with respect to a point, origin.

    Parameters
    ----------
    face : array-like, of shape (N-dim, N-dim)
        The hyperplane we want to know the orientation of
        Do notice that the order in which you provide the points is critical
    origin : array-like, point of shape (N-dim)
        The point to compute the orientation from

    Returns
    -------
    0 if the origin lies in the same hyperplane as face,
    -1 or 1 to indicate left or right orientation

    If two points lie on the same side of the face, the orientation will
    be equal, if they lie on the other side of the face, it will be negated.
    """
    vectors = array(face)
    sign, logdet = slogdet(vectors - origin)
    if logdet < -50:  # assume it to be zero when it's close to zero
        return 0
    return sign


def is_iterable_and_sized(obj):
    return isinstance(obj, Iterable) and isinstance(obj, Sized)


def simplex_volume_in_embedding(vertices) -> float:
    """Calculate the volume of a simplex in a higher dimensional embedding.
    That is: dim > len(vertices) - 1. For example if you would like to know the
    surface area of a triangle in a 3d space.

    This algorithm has not been tested for numerical stability.

    Parameters
    ----------
    vertices : 2D arraylike of floats

    Returns
    -------
    volume : float
        the volume of the simplex with given vertices.

    Raises
    ------
    ValueError
        if the vertices do not form a simplex (for example,
        because they are coplanar, colinear or coincident).
    """
    # Implements http://mathworld.wolfram.com/Cayley-MengerDeterminant.html
    # Modified from https://codereview.stackexchange.com/questions/77593/calculating-the-volume-of-a-tetrahedron

    vertices = asarray(vertices, dtype=float)
    dim = len(vertices[0])
    if dim == 2:
        # Heron's formula
        a, b, c = scipy.spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        return sqrt(s * (s - a) * (s - b) * (s - c))

    # β_ij = |v_i - v_k|²
    sq_dists = scipy.spatial.distance.pdist(vertices, metric="sqeuclidean")

    # Add border while compressed
    num_verts = scipy.spatial.distance.num_obs_y(sq_dists)
    bordered = concatenate((ones(num_verts), sq_dists))

    # Make matrix and find volume
    sq_dists_mat = scipy.spatial.distance.squareform(bordered)

    coeff = -((-2) ** (num_verts - 1)) * factorial(num_verts - 1) ** 2
    vol_square = fast_det(sq_dists_mat) / coeff

    if vol_square < 0:
        if vol_square > -1e-15:
            return 0
        raise ValueError("Provided vertices do not form a simplex")

    return sqrt(vol_square)


class Triangulation:
    """A triangulation object.

    Parameters
    ----------
    coords : 2d array-like of floats
        Coordinates of vertices.

    Attributes
    ----------
    vertices : list of float tuples
        Coordinates of the triangulation vertices.
    simplices : set of integer tuples
        List with indices of vertices forming individual simplices
    vertex_to_simplices : list of sets
        Set of simplices connected to a vertex, the index of the vertex is the
        index of the list.
    hull : set of int
        Exterior vertices

    Raises
    ------
    ValueError
        if the list of coordinates is incorrect or the points do not form one
        or more simplices in the
    """

    def __init__(self, coords):
        if not is_iterable_and_sized(coords):
            raise TypeError("Please provide a 2-dimensional list of points")
        coords = list(coords)
        if not all(is_iterable_and_sized(coord) for coord in coords):
            raise TypeError("Please provide a 2-dimensional list of points")
        if len(coords) == 0:
            raise ValueError("Please provide at least one simplex")
            # raise now because otherwise the next line will raise a less

        dim = len(coords[0])
        if any(len(coord) != dim for coord in coords):
            raise ValueError("Coordinates dimension mismatch")

        if dim == 1:
            raise ValueError("Triangulation class only supports dim >= 2")

        if len(coords) < dim + 1:
            raise ValueError("Please provide at least one simplex")

        coords = list(map(tuple, coords))
        vectors = subtract(coords[1:], coords[0])
        if matrix_rank(vectors) < dim:
            raise ValueError(
                "Initial simplex has zero volumes "
                "(the points are linearly dependent)"
            )

        self.vertices = list(coords)
        self.simplices = set()
        # initialise empty set for each vertex
        self.vertex_to_simplices = [set() for _ in coords]

        # find a Delaunay triangulation to start with, then we will throw it
        # away and continue with our own algorithm
        initial_tri = scipy.spatial.Delaunay(coords)
        for simplex in initial_tri.simplices:
            self.add_simplex(simplex)

    def delete_simplex(self, simplex):
        simplex = tuple(sorted(simplex))
        self.simplices.remove(simplex)
        for vertex in simplex:
            self.vertex_to_simplices[vertex].remove(simplex)

    def add_simplex(self, simplex):
        simplex = tuple(sorted(simplex))
        self.simplices.add(simplex)
        for vertex in simplex:
            self.vertex_to_simplices[vertex].add(simplex)

    def get_vertices(self, indices):
        return [self.get_vertex(i) for i in indices]

    def get_vertex(self, index):
        if index is None:
            return None
        return self.vertices[index]

    def get_reduced_simplex(self, point, simplex, eps=1e-8) -> list:
        """Check whether vertex lies within a simplex.

        Returns
        -------
        vertices : list of ints
            Indices of vertices of the simplex to which the vertex belongs.
            An empty list indicates that the vertex is outside the simplex.
        """
        # XXX: in the end we want to lose this method
        if len(simplex) != self.dim + 1:
            # We are checking whether point belongs to a face.
            simplex = self.containing(simplex).pop()
        x0 = array(self.vertices[simplex[0]])
        vectors = array(self.get_vertices(simplex[1:])) - x0
        alpha = solve(vectors.T, point - x0)
        if any(alpha < -eps) or sum(alpha) > 1 + eps:
            return []

        result = [i for i, a in enumerate(alpha, 1) if a > eps]
        if sum(alpha) < 1 - eps:
            result.insert(0, 0)

        return [simplex[i] for i in result]

    def point_in_simplex(self, point, simplex, eps=1e-8):
        vertices = self.get_vertices(simplex)
        return point_in_simplex(point, vertices, eps)

    def locate_point(self, point):
        """Find to which simplex the point belongs.

        Return indices of the simplex containing the point.
        Empty tuple means the point is outside the triangulation
        """
        for simplex in self.simplices:
            if self.point_in_simplex(point, simplex):
                return simplex
        return ()

    @property
    def dim(self):
        return len(self.vertices[0])

    def faces(self, dim=None, simplices=None, vertices=None):
        """Iterator over faces of a simplex or vertex sequence."""
        if dim is None:
            dim = self.dim

        if simplices is not None and vertices is not None:
            raise ValueError("Only one of simplices and vertices is allowed.")
        if vertices is not None:
            vertices = set(vertices)
            simplices = chain(*(self.vertex_to_simplices[i] for i in vertices))
            simplices = set(simplices)
        elif simplices is None:
            simplices = self.simplices

        faces = (face for tri in simplices for face in combinations(tri, dim))

        if vertices is not None:
            return (face for face in faces if all(i in vertices for i in face))
        else:
            return faces

    def containing(self, face):
        """Simplices containing a face."""
        return set.intersection(*(self.vertex_to_simplices[i] for i in face))

    def _extend_hull(self, new_vertex, eps=1e-8):
        # count multiplicities in order to get all hull faces
        multiplicities = Counter(face for face in self.faces())
        hull_faces = [face for face, count in multiplicities.items() if count == 1]

        # compute the center of the convex hull, this center lies in the hull
        # we do not really need the center, we only need a point that is
        # guaranteed to lie strictly within the hull
        hull_points = self.get_vertices(self.hull)
        pt_center = average(hull_points, axis=0)

        pt_index = len(self.vertices)
        self.vertices.append(new_vertex)

        new_simplices = set()
        for face in hull_faces:
            # do orientation check, if orientation is the same, it lies on
            # the same side of the face, otherwise, it lies on the other
            # side of the face
            pts_face = tuple(self.get_vertices(face))
            orientation_inside = orientation(pts_face, pt_center)
            orientation_new_point = orientation(pts_face, new_vertex)
            if orientation_inside == -orientation_new_point:
                # if the orientation of the new vertex is zero or directed
                # towards the center, do not add the simplex
                simplex = (*face, pt_index)
                if not self._simplex_is_almost_flat(simplex):
                    self.add_simplex(simplex)
                    new_simplices.add(simplex)

        if len(new_simplices) == 0:
            # We tried to add an internal point, revert and raise.
            for tri in self.vertex_to_simplices[pt_index]:
                self.simplices.remove(tri)
            del self.vertex_to_simplices[pt_index]
            del self.vertices[pt_index]
            raise ValueError("Candidate vertex is inside the hull.")

        return new_simplices

    def circumscribed_circle(self, simplex, transform):
        """Compute the center and radius of the circumscribed circle of a simplex.

        Parameters
        ----------
        simplex : tuple of ints
            the simplex to investigate

        Returns
        -------
        tuple (center point, radius)
            The center and radius of the circumscribed circle
        """
        pts = dot(self.get_vertices(simplex), transform)
        return circumsphere(pts)

    def point_in_cicumcircle(self, pt_index, simplex, transform):
        # return self.fast_point_in_circumcircle(pt_index, simplex, transform)
        eps = 1e-8

        center, radius = self.circumscribed_circle(simplex, transform)
        pt = dot(self.get_vertices([pt_index]), transform)[0]

        return norm(center - pt) < (radius * (1 + eps))

    @property
    def default_transform(self):
        return eye(self.dim)

    def bowyer_watson(self, pt_index, containing_simplex=None, transform=None):
        """Modified Bowyer-Watson point adding algorithm.

        Create a hole in the triangulation around the new point,
        then retriangulate this hole.

        Parameters
        ----------
        pt_index: number
            the index of the point to inspect

        Returns
        -------
        deleted_simplices : set of tuples
            Simplices that have been deleted
        new_simplices : set of tuples
            Simplices that have been added
        """
        queue = set()
        done_simplices = set()

        transform = self.default_transform if transform is None else transform

        if containing_simplex is None:
            queue.update(self.vertex_to_simplices[pt_index])
        else:
            queue.add(containing_simplex)

        bad_triangles = set()

        while len(queue):
            simplex = queue.pop()
            done_simplices.add(simplex)

            if self.point_in_cicumcircle(pt_index, simplex, transform):
                self.delete_simplex(simplex)
                todo_points = set(simplex)
                bad_triangles.add(simplex)

                # Get all simplices that share at least a point with the simplex
                neighbors = self.get_neighbors_from_vertices(todo_points)
                # Filter out the already evaluated simplices
                neighbors = neighbors - done_simplices
                neighbors = self.get_face_sharing_neighbors(neighbors, simplex)
                queue.update(neighbors)

        faces = list(self.faces(simplices=bad_triangles))

        multiplicities = Counter(face for face in faces)
        hole_faces = [face for face in faces if multiplicities[face] < 2]

        for face in hole_faces:
            if pt_index not in face:
                simplex = (*face, pt_index)
                if not self._simplex_is_almost_flat(simplex):
                    self.add_simplex(simplex)

        new_triangles = self.vertex_to_simplices[pt_index]
        return bad_triangles - new_triangles, new_triangles - bad_triangles

    def _simplex_is_almost_flat(self, simplex):
        return self._relative_volume(simplex) < 1e-8

    def _relative_volume(self, simplex):
        """Compute the volume of a simplex divided by the average (Manhattan)
        distance of its vertices. The advantage of this is that the relative
        volume is only dependent on the shape of the simplex and not on the
        absolute size. Due to the weird scaling, the only use of this method
        is to check that a simplex is almost flat."""
        vertices = array(self.get_vertices(simplex))
        vectors = vertices[1:] - vertices[0]
        average_edge_length = mean(np_abs(vectors))
        return self.volume(simplex) / (average_edge_length**self.dim)

    def add_point(self, point, simplex=None, transform=None):
        """Add a new vertex and create simplices as appropriate.

        Parameters
        ----------
        point : float vector
            Coordinates of the point to be added.
        transform : N*N matrix of floats
            Multiplication matrix to apply to the point (and neighbouring
            simplices) when running the Bowyer Watson method.
        simplex : tuple of ints, optional
            Simplex containing the point. Empty tuple indicates points outside
            the hull. If not provided, the algorithm costs O(N), so this should
            be used whenever possible.
        """
        point = tuple(point)
        if simplex is None:
            simplex = self.locate_point(point)

        actual_simplex = simplex
        self.vertex_to_simplices.append(set())

        if not simplex:
            temporary_simplices = self._extend_hull(point)

            pt_index = len(self.vertices) - 1
            deleted_simplices, added_simplices = self.bowyer_watson(
                pt_index, transform=transform
            )

            deleted = deleted_simplices - temporary_simplices
            added = added_simplices | (temporary_simplices - deleted_simplices)
            return deleted, added
        else:
            reduced_simplex = self.get_reduced_simplex(point, simplex)
            if not reduced_simplex:
                self.vertex_to_simplices.pop()  # revert adding vertex
                raise ValueError("Point lies outside of the specified simplex.")
            else:
                simplex = reduced_simplex

        if len(simplex) == 1:
            self.vertex_to_simplices.pop()  # revert adding vertex
            raise ValueError("Point already in triangulation.")
        else:
            pt_index = len(self.vertices)
            self.vertices.append(point)
            return self.bowyer_watson(pt_index, actual_simplex, transform)

    def volume(self, simplex):
        prefactor = factorial(self.dim)
        vertices = array(self.get_vertices(simplex))
        vectors = vertices[1:] - vertices[0]
        return float(abs(fast_det(vectors)) / prefactor)

    def volumes(self):
        return [self.volume(sim) for sim in self.simplices]

    def reference_invariant(self):
        """vertex_to_simplices and simplices are compatible."""
        for vertex in range(len(self.vertices)):
            if any(vertex not in tri for tri in self.vertex_to_simplices[vertex]):
                return False
        for simplex in self.simplices:
            if any(simplex not in self.vertex_to_simplices[pt] for pt in simplex):
                return False
        return True

    def vertex_invariant(self, vertex):
        """Simplices originating from a vertex don't overlap."""
        raise NotImplementedError

    def get_neighbors_from_vertices(self, simplex):
        return set.union(*[self.vertex_to_simplices[p] for p in simplex])

    def get_face_sharing_neighbors(self, neighbors, simplex):
        """Keep only the simplices sharing a whole face with simplex."""
        return {
            simpl for simpl in neighbors if len(set(simpl) & set(simplex)) == self.dim
        }  # they share a face

    def get_simplices_attached_to_points(self, indices):
        # Get all simplices that share at least a point with the simplex
        neighbors = self.get_neighbors_from_vertices(indices)
        return self.get_face_sharing_neighbors(neighbors, indices)

    def get_opposing_vertices(self, simplex):
        if simplex not in self.simplices:
            raise ValueError("Provided simplex is not part of the triangulation")
        neighbors = self.get_simplices_attached_to_points(simplex)

        def find_opposing_vertex(vertex):
            # find the simplex:
            simp = next((x for x in neighbors if vertex not in x), None)
            if simp is None:
                return None
            opposing = set(simp) - set(simplex)
            assert len(opposing) == 1
            return opposing.pop()

        result = tuple(find_opposing_vertex(v) for v in simplex)
        return result

    @property
    def hull(self):
        """Compute hull from triangulation.

        Parameters
        ----------
        check : bool, default: True
            Whether to raise an error if the computed hull is different from
            stored.

        Returns
        -------
        hull : set of int
            Vertices in the hull.
        """
        counts = Counter(self.faces())
        if any(i > 2 for i in counts.values()):
            raise RuntimeError(
                "Broken triangulation, a (N-1)-dimensional"
                " appears in more than 2 simplices."
            )

        hull = {point for face, count in counts.items() if count == 1 for point in face}
        return hull

    def convex_invariant(self, vertex):
        """Hull is convex."""
        raise NotImplementedError
