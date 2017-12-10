from collections import defaultdict, Counter
from itertools import combinations, chain

import numpy as np
from scipy import linalg


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

    def _extend_hull(self, new_vertex, eps=1e-8):
        hull_faces = list(self.faces(vertices=self.hull))
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

        pt_index = len(self.vertices)
        self.vertices.append(new_vertex)
        for face in hull_faces:
            if all(i in new_vertices for i in face):
                self.add_simplex(face + (pt_index,))

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

    def add_point(self, point, simplex=None):
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
            self._extend_hull(point)
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
            self.add_point_inside_simplex(point, simplex)
        else:
            self.add_point_on_face(point, simplex)

    def add_point_inside_simplex(self, point, simplex):
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

        return(new)

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
                self.add_simplex(others + opposing + (pt_index,))

    def flip(self, face):
        """Flip the face shared between several simplices."""
        simplices = self.containing(face)

        new_face = tuple(set.union(*(set(tri) for tri in simplices))
                         - set(face))
        if len(new_face) + len(face) != self.dim + 2:
            # TODO: is this condition correct for arbitraty face dimension in
            # d>2?
            raise RuntimeError("face has too few or too many neighbors.")

        new_simplices = [others + new_face for others in
                         combinations(face, len(face) - 1)]

        volume_new = sum(self.volume(tri) for tri in new_simplices)
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
