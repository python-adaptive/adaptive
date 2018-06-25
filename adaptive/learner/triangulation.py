from collections import defaultdict, Counter
from itertools import combinations, chain

import numpy as np
from scipy import linalg
"""Add a new vertex and create simplices as appropriate.

        Parameters
        ----------
        point : float vector
            Coordinates of the point to be added.
        simplex : tuple of ints, optional
            Simplex containing the point. Empty tuple indicates points outside
            the hull. If not provided, the algorithm costs O(N), so this should
            be used whenever possible.
        """

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
        vectors = np.array([self.vertices[i] for i in simplex[1:]]) - x0
        alpha = np.linalg.solve(vectors.T, point - x0)
        if any(alpha < -eps) or sum(alpha) > 1 + eps:
            return []
        result = (
            ([0] if sum(alpha)  < 1 - eps else [])
            + [i + 1 for i, a in enumerate(alpha) if a > eps]
        )
        return [simplex[i] for i in result]

    def locate_point(self, point):
        """Find to which simplex the point belongs.

        Return indices of the simplex containing the point.
        Empty tuple means the point is outside the triangulation
        """
        for simplex in self.simplices:
            face = self.point_in_simplex(point, simplex)
            if face:
                return face

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

    def _extend_hull(self, new_vertex, allow_flip=False, eps=1e-8):
        hull_faces = list(self.faces(vertices=self.hull))
        # notice that this also includes interior faces, to remove these we
        # count multiplicities
        multiplicities = Counter(face for face in hull_faces)
        hull_faces = [face for face in hull_faces if multiplicities.get(face) < 2]

        decomp = []
        for face in hull_faces:
            coords = np.array([self.vertices[i] for i in face]) - new_vertex
            decomp.append(linalg.lu_factor(coords.T))
        shifted = [self.vertices[vertex] - np.array(new_vertex)
                   for vertex in self.hull]

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
        hull_points = []
        for p in self.hull:
            hull_points.append(self.vertices[p])
        pt_center = np.average(hull_points, axis=0)


        pt_index = len(self.vertices)
        self.vertices.append(new_vertex)
        faces_to_check = set()
        for face in hull_faces:
            if all(i in new_vertices for i in face) or True:
                # do orientation check, if orientation is the same, it lies on
                # the same side of the face, otherwise, it lies on the other
                # side of the face
                pts_face = tuple(self.vertices[i] for i in face)
                orientation_inside = orientation(pts_face, pt_center)
                orientation_new_point = orientation(pts_face, new_vertex)
                if orientation_inside == -orientation_new_point:
                    # if the orientation of the new vertex is zero or directed
                    # towards the center, do not add the simplex
                    self.add_simplex(face + (pt_index,))
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

        if allow_flip:
            for face in faces_to_check:
                self.flip_if_needed(face)



    def circumscribed_circle(self, simplex):
        """
        Compute the centre and radius of the circumscribed circle of a simplex
        :param simplex: the simplex to investigate
        :return: tuple (centre point, radius)
        """
        # Modified from http://mathworld.wolfram.com/Circumsphere.html
        mat = []
        for i in simplex:
            pt = self.vertices[i]
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
        radius = np.linalg.norm(np.subtract(center, x0))

        for i in simplex:
            if radius < 1e6 and abs(np.linalg.norm(center - np.array(self.vertices[i])) - radius) > 1e-8:
                raise RuntimeError("Error in finding Circumscribed Circle")

        return tuple(center), radius


    def flip_if_needed(self, face):
        """
        check if face needs to be flipped, and flip if this is the case
        :param face: a face
        """
        do_flip = False
        simplices = self.containing(face)
        if len(simplices) < 2:
            return  # we are at a border, do not flip

        simplex = simplices.pop()

        centre, radius = self.circumscribed_circle(simplex)

        other_points = set.union(set(simplex), *simplices) - set(simplex)
        # TODO if a flip would create a coplanar simplex, do not flip,
        # or better even, do a special flip that makes the triangulation the best it can be
        # see http://www.kiv.zcu.cz/site/documents/verejne/vyzkum/publikace/technicke-zpravy/2002/tr-2002-02.pdf
        # for inspiration

        for i in other_points:
            pt = np.array(self.vertices[i])
            # if Delaunay flip condition is met, do a flip
            if np.linalg.norm(centre - pt) < radius:
                try:
                    self.flip(face)
                except RuntimeError as e:
                    pass
                return

    def add_point(self, point, simplex=None, allow_flip=False):
        """Add a new vertex and create simplices as appropriate.

        Parameters
        ----------
        point : float vector
            Coordinates of the point to be added.
        simplex : tuple of ints, optional
            Simplex containing the point. Empty tuple indicates points outside
            the hull. If not provided, the algorithm costs O(N), so this should
            be used whenever possible.
        """
        if simplex is None:
            simplex = self.locate_point(point)

        if not simplex:
            self._extend_hull(point, allow_flip=allow_flip)
            return
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
        elif len(simplex) == self.dim + 1:
            self.add_point_inside_simplex(point, simplex, allow_flip)
        else:
            self.add_point_on_face(point, simplex, allow_flip)

    def add_point_inside_simplex(self, point, simplex, allow_flip=False):
        if len(self.point_in_simplex(point, simplex)) != self.dim + 1:
            raise ValueError("Vertex is not inside simplex")
        pt_index = len(self.vertices)
        self.vertices.append(point)
        self.delete_simplex(simplex)

        new = []
        for others in combinations(simplex, len(simplex) - 1):
            tri = others + (pt_index,)
            self.add_simplex(tri)
            new.append(tri)
            # TODO do a check for the Flip condition
            if allow_flip:
                self.flip_if_needed(others)

        return(new)

    def add_point_on_face(self, point, face, allow_flip=False):
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
                self.add_simplex(others + opposing + (pt_index,))
                # TODO do a check for the Flip condition
                if allow_flip:
                    self.flip_if_needed(others + opposing)

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
        vectors = np.array([self.vertices[i] for i in simplex[1:]])
        return abs(np.linalg.det(vectors
                                 - self.vertices[simplex[0]])) / prefactor

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
