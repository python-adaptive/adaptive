from collections import defaultdict, Counter
from itertools import combinations, chain

import numpy as np
from scipy import linalg
import math

def fast_norm(v):
    # notice this method can be even more optimised
    return math.sqrt(np.dot(v,v))

def fast_norm_2d(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

def fast_2d_point_in_simplex(point, simplex, eps=1e-8):
    (p0x, p0y), (p1x, p1y), (p2x, p2y) = simplex
    px, py = point

    area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)

    s = 1 / (2 * area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
    if s < -eps or s > 1+eps:
        return False
    t = 1 / (2 * area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)

    return (t >= -eps) and (s + t <= 1+eps)

# WIP
# def fast_3d_point_in_simplex(point, simplex, eps=1e-8):
#     p0x, p0y, p0z = simplex[0]
#     p1x, p1y, p1z = simplex[1]
#     p2x, p2y, p2z = simplex[2]
#     p3x, p3y, p3z = simplex[3]
#     px, py, pz = point
#
#     Area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)
#
#     s = 1 / (2 * Area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
#     if s < -eps or s > 1+eps:
#         return
#     t = 1 / (2 * Area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)
#     if t < -eps or s+t > 1+eps:
#         return
#     u = 0
#
#     return (s >= 0) and (u >= 0) and (s + t + u<= 1)


def fast_2d_circumcircle(points):
    """
    Compute the centre and radius of the circumscribed circle of a simplex
    :param points: the triangle to investigate
    :return: tuple (centre point, radius)
    """
    points = np.array(points)
    # transform to relative coordinates
    pts = points[1:] - points[0]

    (x1, y1), (x2, y2) = pts
    # compute the length squared
    l1 = x1*x1 + y1*y1
    l2 = x2*x2 + y2*y2

    # compute some determinants
    dx = + l1 * y2 - l2 * y1
    dy = - l1 * x2 + l2 * x1
    aa = + x1 * y2 - x2 * y1
    a = 2 * aa

    # compute center
    x = dx/a
    y = dy/a
    radius = math.sqrt(x*x + y*y) # radius = norm([x, y])

    return (x + points[0][0], y + points[0][1]), radius


def fast_3d_circumcircle(points):
    """
    Compute the centre and radius of the circumscribed circle of a simplex
    :param points: the simplex to investigate
    :return: tuple (centre point, radius)
    """
    points = np.array(points)
    pts = points[1:] - points[0]

    l1, l2, l3 = np.dot(pts.T, pts) # length squared
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = pts

    # Compute some determinants:
    dx = + l1 * (y2 * z3 - z2 * y3) - l2 * (y1 * z3 - z1 * y3) + l3 * (y1 * z2 - z1 * y2)
    dy = + l1 * (x2 * z3 - z2 * x3) - l2 * (x1 * z3 - z1 * x3) + l3 * (x1 * z2 - z1 * x2)
    dz = + l1 * (x2 * y3 - y2 * x3) - l2 * (x1 * y3 - y1 * x3) + l3 * (x1 * y2 - y1 * x2)
    aa = + x1 * (y2 * z3 - z2 * y3) - x2 * (y1 * z3 - z1 * y3) + x3 * (y1 * z2 - z1 * y2)
    a = 2*aa

    center = [dx/a, -dy/a, dz/a]
    radius = fast_norm(center)
    center = np.add(center, points[0])

    return tuple(center), radius


def orientation(face, origin):
    """Compute the orientation of the face with respect to a point, origin

    Parameters
    ----------
    Face : array-like, of shape (N-dim, N-dim)
        The hyperplane we want to know the orientation of
        Do notice that the order in which you provide the points is critical
    Origin : array-like, point of shape (N-dim)
        The point to compute the orientation from

    Returns
    -------
      0 if the origin lies in the same hyperplane as face,
      -1 or 1 to indicate left or right orientation

      If two points lie on the same side of the face, the orientation will
      be equal, if they lie on the other side of the face, it will be negated.
    """
    vectors = np.array(face)
    sign, logdet = np.linalg.slogdet(vectors - origin)
    if logdet < -50:  # assume it to be zero when it's close to zero
        return 0
    return sign


class Triangulation:
    def __init__(self, coords):
        """A triangulation object.

        Parameters
        ----------
        coords : 2d array-like of floats
            Coordinates of vertices of the first simplex.

        Attributes
        ----------
        vertices : list of float tuples
            Coordinates of the triangulation vertices.
        simplices : set of integer tuples
            List with indices of vertices forming individual simplices
        vertex_to_simplices : dict int → set
            Mapping from vertex index to the set of simplices containing that
            vertex.
        hull : set of int
            Exterior vertices
        """
        dim = len(coords[0])
        if any(len(coord) != dim for coord in coords):
            raise ValueError("Coordinates dimension mismatch")

        if len(coords) != dim + 1:
            raise ValueError("Can only add one simplex on initialization")

        self.vertices = list(coords)
        self.simplices = set()
        self.vertex_to_simplices = defaultdict(set)
        self.add_simplex(range(len(self.vertices)))
        self.hull = set(range(len(coords)))

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
            return [self.vertices[i] for i in indices]
            return

    def point_in_simplex(self, point, simplex, eps=1e-8):
        """Check whether vertex lies within a simplex.

        Returns
        -------
        vertices : list of ints
            Indices of vertices of the simplex to which the vertex belongs. None
            indicates that the vertex is outside the simplex
        """
        if len(simplex) != self.dim + 1:
            # We are checking whether point belongs to a face.
            simplex = self.containing(simplex).pop()
        x0 = np.array(self.vertices[simplex[0]])
        vectors = np.array(self.get_vertices(simplex[1:])) - x0
        alpha = np.linalg.solve(vectors.T, point - x0)
        if any(alpha < -eps) or sum(alpha) > 1 + eps:
            return []

        result = [i for i, a in enumerate(alpha, 1) if a > eps]
        if sum(alpha) < 1 - eps:
            result.insert(0, 0)

        return [simplex[i] for i in result]

    def fast_point_in_simplex(self, point, simplex, eps=1e-8):
        if self.dim == 2:
            return fast_2d_point_in_simplex(point, self.get_vertices(simplex), eps)
        elif self.dim == 3:
            return self.point_in_simplex(point, simplex, eps)
        else:
            return self.point_in_simplex(point, simplex, eps)

    def locate_point(self, point):
        """Find to which simplex the point belongs.

        Return indices of the simplex containing the point.
        Empty tuple means the point is outside the triangulation
        """
        for simplex in self.simplices:
            if self.fast_point_in_simplex(point, simplex):
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

        faces = (face for tri in simplices
                 for face in combinations(tri, dim))

        if vertices is not None:
            return (face for face in faces if all(i in vertices for i in face))
        else:
            return faces

    def containing(self, face):
        """Simplices containing a face."""
        return set.intersection(*(self.vertex_to_simplices[i] for i in face))

    def _extend_hull(self, new_vertex, eps=1e-8):
        hull_faces = list(self.faces(vertices=self.hull))
        # notice that this also includes interior faces, to remove these we
        # count multiplicities
        multiplicities = Counter(face for face in hull_faces)
        hull_faces = [face for face in hull_faces if multiplicities[face] < 2]

        decomp = []
        for face in hull_faces:
            coords = np.array(self.get_vertices(face)) - new_vertex
            decomp.append(linalg.lu_factor(coords.T))

        shifted = np.subtract(self.get_vertices(self.hull), new_vertex)

        new_vertices = set()
        for coord, index in zip(shifted, self.hull):
            good = True
            for face, factored in zip(hull_faces, decomp):
                if index in face:
                    continue
                alpha = linalg.lu_solve(factored, coord)
                if all(alpha > eps):
                    good = False
                    break
            if good:
                new_vertices.add(index)

        # compute the center of the convex hull, this center lies in the hull
        # we do not really need the center, we only need a point that is
        # guaranteed to lie strictly within the hull
        hull_points = self.get_vertices(self.hull)
        pt_center = np.average(hull_points, axis=0)


        pt_index = len(self.vertices)
        self.vertices.append(new_vertex)
        faces_to_check = set()
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
                self.add_simplex((*face, pt_index))
                faces_to_check.add(face)

        multiplicities = Counter(face for face in
                                self.faces(vertices=new_vertices | {pt_index})
                                if pt_index in face)

        if all(i == 2 for i in multiplicities.values()):
            # We tried to add an internal point, revert and raise.
            for tri in self.vertex_to_simplices[pt_index]:
                self.simplices.remove(tri)
            del self.vertex_to_simplices[pt_index]
            del self.vertices[pt_index]
            raise ValueError("Candidate vertex is inside the hull.")

        self.hull.add(pt_index)
        self.hull = self.compute_hull(vertices=self.hull, check=False)


    def circumscribed_circle(self, simplex, transform):
        """
        Compute the centre and radius of the circumscribed circle of a simplex
        :param simplex: the simplex to investigate
        :return: tuple (centre point, radius)
        """
        pts = np.dot(self.get_vertices(simplex), transform)
        if self.dim == 2:
            return fast_2d_circumcircle(pts)
        if self.dim == 3:
            return fast_3d_circumcircle(pts)

        # Modified from http://mathworld.wolfram.com/Circumsphere.html
        mat = []
        for pt in pts:
            length_squared = np.sum(np.square(pt))
            row = np.array([length_squared, *pt, 1])
            mat.append(row)

        center = []
        for i in range(1, len(simplex)):
            r = np.delete(mat, i, 1)
            factor = (-1) ** (i+1)
            center.append(factor * np.linalg.det(r))

        a = np.linalg.det(np.delete(mat, 0, 1))
        center = [x / (2*a) for x in center]

        x0 = self.vertices[next(iter(simplex))]
        vec = np.subtract(center, x0)
        radius = fast_norm(vec)

        return tuple(center), radius


    def point_in_cicumcircle(self, pt_index, simplex, transform):
        # return self.fast_point_in_circumcircle(pt_index, simplex, transform)
        eps = 1e-8

        center, radius = self.circumscribed_circle(simplex, transform)
        pt = np.dot(self.get_vertices([pt_index]), transform)[0]


        return np.linalg.norm(center - pt) < (radius * (1 + eps))

    # WIP: currently this is slower than computing the circumcircle
    # def fast_point_in_circumcircle(self, pt_index, simplex, transform):
    #     # Construct the matrix
    #     eps = 1e-10
    #     indices = simplex + (pt_index,)
    #     original_points = self.get_vertices(indices)
    #     points = np.dot(original_points, transform)
    #     l_squared = np.sum(np.square(points), axis=1)
    #
    #     M = np.array([*np.transpose(points), l_squared, np.ones(l_squared.shape)], dtype=float)
    #
    #     # Compute the determinant
    #     det = np.linalg.det(M)
    #     if np.abs(det) < eps:
    #         return True
    #
    #     M2 = [*np.transpose(points[:-1]), np.ones(len(simplex))]
    #     det_inside = np.linalg.det(M2)
    #
    #     return np.sign(det) == np.sign(det_inside)

    @property
    def default_transform(self):
        return np.eye(self.dim)

    def bowyer_watson(self, pt_index, containing_simplex=None, transform=None):
        """
        Modified Bowyer-Watson point adding algorithm

        Create a hole in the triangulation around the new point, then retriangulate this hole.

        :param pt_index: the index of the point to inspect
        :return: deleted_simplices, new_simplices
        """
        queue = set()
        done_simplices = set()

        transform = self.default_transform if transform is None else transform

        if containing_simplex is None:
            queue.update(self.vertex_to_simplices[pt_index])
        else:
            queue.add(containing_simplex)

        done_points = {pt_index}

        bad_triangles = set()

        while len(queue):
            simplex = queue.pop()
            done_simplices.add(simplex)

            if self.point_in_cicumcircle(pt_index, simplex, transform):
                self.delete_simplex(simplex)
                todo_points = set(simplex) - done_points
                done_points.update(simplex)

                if len(todo_points):
                    neighbours = set.union(*[self.vertex_to_simplices[p] for p in todo_points])
                    queue.update(neighbours - done_simplices)

                bad_triangles.add(simplex)

        faces = list(self.faces(simplices=bad_triangles))

        multiplicities = Counter(face for face in faces)
        hole_faces = [face for face in faces if multiplicities[face] < 2]

        for face in hole_faces:
            if pt_index not in face:
                if self.volume((*face, pt_index)) < 1e-8:
                    continue
                self.add_simplex((*face, pt_index))

        return bad_triangles, self.vertex_to_simplices[pt_index]

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

        if not simplex:
            self._extend_hull(point)

            pt_index = len(self.vertices) - 1
            # self.bowyer_watson(pt_index)
            return self.bowyer_watson(pt_index, transform=transform)
        else:
            reduced_simplex = self.point_in_simplex(point, simplex)
            if not reduced_simplex:
                raise ValueError(
                    'Point lies outside of the specified simplex.'
                )
            else:
                simplex = reduced_simplex

        if len(simplex) == 1:
            raise ValueError("Point already in triangulation.")
        else:
            pt_index = len(self.vertices)
            self.vertices.append(point)
            return self.bowyer_watson(pt_index, actual_simplex, transform)

    def add_point_inside_simplex(self, point, simplex):
        if len(self.point_in_simplex(point, simplex)) != self.dim + 1:
            raise ValueError("Vertex is not inside simplex")
        pt_index = len(self.vertices)
        self.vertices.append(point)
        self.delete_simplex(simplex)

        new = []
        for others in combinations(simplex, len(simplex) - 1):
            tri = (*others, pt_index)
            self.add_simplex(tri)
            new.append(tri)

        return new

    def add_point_on_face(self, point, face):
        pt_index = len(self.vertices)
        self.vertices.append(point)

        simplices = self.containing(face)
        if (set(self.point_in_simplex(point, next(iter(simplices))))
                != set(face)):

            raise ValueError("Vertex does not lie on the face.")

        if all(pt in self.hull for pt in face):
            self.hull.add(pt_index)
        for simplex in simplices:
            self.delete_simplex(simplex)
            opposing = tuple(pt for pt in simplex if pt not in face)

            for others in combinations(face, len(face) - 1):
                self.add_simplex((*others, *opposing, pt_index))

    def flip(self, face):
        """Flip the face shared between several simplices."""
        simplices = self.containing(face)

        new_face = tuple(set.union(*(set(tri) for tri in simplices))
                         - set(face))
        if len(new_face) + len(face) != self.dim + 2:
            # TODO: is this condition correct for arbitrary face dimension in
            # d>2?
            raise RuntimeError("face has too few or too many neighbors.")

        new_simplices = [others + new_face for others in
                         combinations(face, len(face) - 1)]

        new_volumes = [self.volume(tri) for tri in new_simplices]
        volume_new = sum(new_volumes)

        # do not allow creation of zero-volume
        if any([(v < 1e-10) for v in new_volumes]):
            raise RuntimeError(
                "face cannot be flipped without creating a zero volume simplex"
                "the corner points are coplanar"
            )

        volume_was = sum(self.volume(tri) for tri in simplices)

        if not np.allclose(volume_was, volume_new):
            raise RuntimeError(
                "face cannot be flipped without breaking the triangulation."
            )

        for simplex in new_simplices:
            self.add_simplex(simplex)

        for simplex in simplices:
            self.delete_simplex(simplex)

    def volume(self, simplex):
        prefactor = np.math.factorial(self.dim)
        vertices = np.array(self.get_vertices(simplex))
        vectors = vertices[1:] - vertices[0]
        return abs(np.linalg.det(vectors)) / prefactor

    def reference_invariant(self):
        """vertex_to_simplices and simplices are compatible."""
        for vertex in range(len(self.vertices)):
            if any(vertex not in tri
                   for tri in self.vertex_to_simplices[vertex]):
                return False
        for simplex in self.simplices:
            if any(simplex not in self.vertex_to_simplices[pt]
                   for pt in simplex):
                return False
        return True

    def vertex_invariant(self, vertex):
        """Simplices originating from a vertex don't overlap."""
        raise NotImplementedError

    def compute_hull(self, vertices=None, check=True):
        """Recompute hull from triangulation.

        Parameters
        ----------
        vertices : set of int
        check : bool, default True
            Whether to raise an error if the computed hull is different from
            stored.

        Returns
        -------
        hull : set of int
            Vertices in the hull.
        """
        counts = Counter(self.faces())
        if any(i > 2 for i in counts.values()):
            raise RuntimeError("Broken triangulation, a d-1 dimensional face "
                               "appears in more than 2 simplices.")

        hull = set(point for face, count in counts.items() if count == 1
                   for point in face)

        if check and self.hull != hull:
            raise RuntimeError("Incorrect hull value.")

        return hull

    def convex_invariant(self, vertex):
        """Hull is convex."""
        raise NotImplementedError
