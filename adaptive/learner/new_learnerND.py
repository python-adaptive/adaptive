from math import sqrt
import itertools
from collections.abc import Iterable

import numpy as np
import scipy.spatial
import scipy.interpolate
from sortedcontainers import SortedList, SortedDict

from adaptive.learner.base_learner import BaseLearner
from adaptive.learner.triangulation import (
    Triangulation,
    simplex_volume_in_embedding,
    circumsphere,
    point_in_simplex,
)
from adaptive.notebook_integration import ensure_holoviews


class Domain:
    def insert_points(self, subdomain, n):
        """Insert 'n' points into 'subdomain'.

        Returns
        -------
        affected_subdomains : Iterable of subdomains
            If some points were added on the boundary of 'subdomain'
            then they will also have been added to the neighboring
            subdomains.
        """

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

    def transform(self, x):
        "Transform 'x' to the unit hypercube"

    def neighbors(self, subdomain, n=1):
        "Return all neighboring subdomains up to degree 'n'."

    def subdomains(self):
        "Return all the subdomains in the domain."

    def clear_subdomains(self):
        """Remove all points from the interior of subdomains.

        Returns
        -------
        subdomains : the subdomains who's interior points were removed
        """

    def volume(self, subdomain):
        "Return the volume of a subdomain."

    def subvolumes(self, subdomain):
        "Return the volumes of the sub-subdomains."


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
        if _check_membership and subdomain not in self:
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
            m = a + (b - a) / 2
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
        subdomain = (p[i - 1], p[i])

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
            sub_points = self.sub_domains[subdomain]
        except KeyError:
            raise ValueError("{} not in any subdomain".format(x))
        else:
            sub_points.remove(x)
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

    def __contains__(self, subdomain):
        a, b = subdomain
        try:
            ia = self.points.index(a)
            ib = self.points.index(b)
        except ValueError:
            return False
        return ia + 1 == ib

    def transform(self, x):
        a, b = self.bounds
        return (x - a) / (b - a)

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
    face : tuple of int
        If the chosen point was
    """
    if transform is not None:
        simplex = np.dot(simplex, transform)

    # Choose center only if the shape of the simplex is nice,
    # otherwise: the center the longest edge
    center, _radius = circumsphere(simplex)
    if point_in_simplex(center, simplex):
        point = np.average(simplex, axis=0)
        face = ()
    else:
        distances = scipy.spatial.distance.pdist(simplex)
        distance_matrix = scipy.spatial.distance.squareform(distances)
        i, j = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        point = (simplex[i, :] + simplex[j, :]) / 2
        face = (i, j)

    if transform is not None:
        point = np.linalg.solve(transform, point)  # undo the transform

    return point, face


class ConvexHull(Domain):
    """A convex hull domain in $ℝ^N$ (N >=2).

    Subdomains are simplices represented by integer tuples of length (N + 1).
    """

    def __init__(self, hull):
        assert isinstance(hull, scipy.spatial.ConvexHull)

        self.bounds = hull
        self.triangulation = Triangulation(hull.points[hull.vertices])
        # if a subdomain has interior points, then it appears as a key
        # in 'sub_domains' and maps to a 'Triangulation' of the
        # interior of the subdomain. By definition the triangulation
        # is over a simplex, and the first 'ndim + 1' points in the
        # triangulation are the boundary points.
        self.sub_domains = dict()
        self.ndim = self.bounds.points.shape[1]

    @property
    def bounding_box(self):
        hull_points = self.bounds.points[self.bounds.vertices]
        return tuple(zip(hull_points.min(axis=0), hull_points.max(axis=0)))

    def _get_subtriangulation(self, subdomain):
        try:
            subtri = self.sub_domains[subdomain]
        except KeyError:  # No points in the interior of this subdomain yet
            subtri = Triangulation([self.triangulation.vertices[x] for x in subdomain])
            self.sub_domains[subdomain] = subtri
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
            point, face = _choose_point_in_simplex(simplex_vertices)
            points.append(point)
            subtri.add_point(point, largest_simplex)
            # If we chose a point on a face (or edge) of 'subdomain' then we need to
            # add it to the subtriangulations of the neighboring subdomains.
            # This check relies on the fact that the first 'ndim + 1' points in the
            # subtriangulation are the boundary points.
            face = [largest_simplex[i] for i in face]
            if face and all(f < self.ndim + 1 for f in face):
                # Translate vertex indices from subtriangulation to triangulation
                face = [subdomain[f] for f in face]
                # Loop over the simplices that contain 'face', skipping 'subdomain',
                # which was already added above.
                for sd in tri.containing(face):
                    if sd != subdomain:
                        self._get_subtriangulation(sd).add_point(point)
                        affected_subdomains.add(sd)

        return [tuple(p) for p in points], affected_subdomains

    def insert(self, x, *, _check_membership=True):
        # XXX: O(N) in the number of simplices
        affected_subdomains = self.which_subdomains(x)
        if not affected_subdomains:
            raise ValueError("{} is not present in this domain".format(x))
        for subdomain in affected_subdomains:
            subtri = self._get_subtriangulation(subdomain)
            if x in subtri.vertices:  # O(N) in the number of vertices
                raise ValueError("{} exists in a subinterval already".format(x))
            subtri.add_point(x)

        return affected_subdomains

    def remove(self, x):
        # XXX: O(N) in the number of simplices
        affected_subdomains = self.which_subdomains(x)
        for subdomain in affected_subdomains:
            # Check that it's not a vertex of the subdomain
            if any(x == self.triangulation.vertices[i] for i in subdomain):
                raise ValueError("Cannot remove subdomain vertices")
            try:
                subtri = self.sub_domains[subdomain]
            except KeyError:
                raise ValueError("{} not present in any subdomain".format(x))
            else:
                if x not in subtri.vertices:
                    raise ValueError("{} not present in any subdomain".format(x))
                # Rebuild the subtriangulation from scratch
                self.sub_domains[subdomain] = Triangulation(
                    [v for v in subtri.vertices if v != x]
                )

    def split_at(self, x, *, _check_membership=True):
        tri = self.triangulation
        # XXX: O(N) in the number of simplices. As typically 'x' will have been
        #      obtained by 'insert_points' or by calling 'insert_into' we can keep
        #      a hashmap of x→simplex to make this lookup faster and fall back to
        #      'locate_point' otherwise
        simplex = tri.locate_point(x)
        if not simplex:
            raise ValueError("Can only split at points within the domain.")

        old_subdomains, new_subdomains = tri.add_point(x, simplex)

        if _check_membership:
            assert not any(s in self.sub_domains for s in new_subdomains)

        # Re-assign all the interior points of 'old_subdomains' to 'new_subdomains'

        # Keep the interior points as a set, because interior points on a shared face
        # appear in the subtriangulations of both the neighboring simplices, and we
        # don't want those points to appear twice.
        interior_points = set()
        for d in old_subdomains:
            try:
                subtri = self.sub_domains.pop(d)
            except KeyError:
                continue
            else:
                # Get all points in the subtriangulation except the boundary
                # points. Because a subtriangulation is always defined over
                # a simplex, the first ndim + 1 points are the boundary points.
                interior = set(range(self.ndim + 1, len(subtri.vertices)))
                interior = [subtri.vertices[i] for i in interior]
                # Remove 'x' if it is one of the points
                interior = [i for i in interior if i != x]
                interior_points.update(interior)
        for p in interior_points:
            # Try to add 'p' to all the new subdomains. It may belong to more than 1
            # if it lies on a subdomain boundary.
            p_was_added = False
            for subdomain in new_subdomains:
                if tri.point_in_simplex(p, subdomain):
                    try:
                        subtri = self.sub_domains[subdomain]
                    except KeyError:  # No points in this subdomain yet
                        subtri = Triangulation([tri.vertices[i] for i in subdomain])
                        self.sub_domains[subdomain] = subtri
                    subtri.add_point(p)
                    p_was_added = True
            assert (
                p_was_added
            ), "{} was not in the interior of any new simplices".format(x)

        return old_subdomains, new_subdomains

    def which_subdomains(self, x):
        tri = self.triangulation
        # XXX: O(N) in the number of simplices
        subdomains = [s for s in tri.simplices if tri.point_in_simplex(x, s)]
        if not subdomains:
            raise ValueError("{} is not in the domain".format(x))
        return subdomains

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

    def clear_subdomains(self):
        sub_domains = list(self.sub_domains.keys())
        self.sub_domains = dict()
        return sub_domains

    def volume(self, subdomain):
        return self.triangulation.volume(subdomain)

    def subvolumes(self, subdomain):
        try:
            subtri = self.sub_domains[subdomain]
        except KeyError:
            return [self.triangulation.volume(subdomain)]
        else:
            return [subtri.volume(s) for s in subtri.simplices]


class Queue:
    """Priority queue supporting update and removal at arbitrary position.

    Parameters
    ----------
    entries : iterable of (item, priority)
        The initial data in the queue. Providing this is faster than
        calling 'insert' a bunch of times.
    """

    def __init__(self, entries=()):
        self._queue = SortedDict(
            ((priority, n), item) for n, (item, priority) in enumerate(entries)
        )
        # 'self._queue' cannot be keyed only on priority, as there may be several
        # items that have the same priority. To keep unique elements the key
        # will be '(priority, self._n)', where 'self._n' is incremented whenever
        # we add a new element.
        self._n = len(self._queue)
        # To efficiently support updating and removing items if their priority
        # is unknown we have to keep the reverse map of 'self._queue'. Because
        # items may not be hashable we cannot use a SortedDict, so we use a
        # SortedList storing '(item, key)'.
        self._items = SortedList(((v, k) for k, v in self._queue.items()))

    def items(self):
        "Return an iterator over the items in the queue in arbitrary order."
        return reversed(self._queue.values())

    def priorities(self):
        "Return an iterator over the priorities in the queue in arbitrary order."
        return reversed(self._queue)

    def peek(self):
        "Return the item and priority at the front of the queue."
        ((priority, _), item) = self._queue.peekitem()
        return item, priority

    def pop(self):
        "Remove and return the item and priority at the front of the queue."
        (key, item) = self._queue.popitem()
        i = self._items.index((item, key))
        del self._items[i]
        priority, _ = key
        return item, priority

    def insert(self, item, priority):
        "Insert 'item' into the queue with the given priority."
        key = (priority, self._n)
        self._items.add((item, key))
        self._queue[key] = item
        self._n += 1

    def remove(self, item):
        "Remove the 'item' from the queue."
        i = self._items.bisect_left((item, ()))
        should_be, key = self._items[i]
        if item != should_be:
            raise KeyError("item is not in queue")

        del self._queue[key]
        del self._items[i]

    def update(self, item, priority):
        """Update 'item' in the queue to have the given priority.

        Raises
        ------
        KeyError : if 'item' is not in the queue.
        """
        i = self._items.bisect_left((item, ()))
        should_be, key = self._items[i]
        if item != should_be:
            raise KeyError("item is not in queue")

        _, n = key
        new_key = (priority, n)

        del self._queue[key]
        self._queue[new_key] = item
        del self._items[i]
        self._items.add((item, new_key))


class LossFunction:
    @property
    def n_neighbors(self):
        "The maximum degree of neighboring subdomains required."

    def __call__(self, domain, subdomain, data):
        """Return the loss for 'subdomain' given 'data'

        Neighboring subdomains can be obtained with
        'domain.neighbors(subdomain, self.n_neighbors)'.
        """


class DistanceLoss(LossFunction):
    @property
    def n_neighbors(self):
        return 0

    def __call__(self, domain, subdomain, codomain_bounds, data):
        assert isinstance(domain, Interval)
        a, b = subdomain
        ya, yb = data[a], data[b]
        return sqrt((b - a) ** 2 + (yb - ya) ** 2)


class EmbeddedVolumeLoss(LossFunction):
    @property
    def n_neighbors(self):
        return 0

    def __call__(self, domain, subdomain, codomain_bounds, data):
        assert isinstance(domain, ConvexHull)
        xs = [tuple(domain.triangulation.vertices[x]) for x in subdomain]
        ys = [data[x] for x in xs]
        if isinstance(ys[0], Iterable):
            pts = [(*x, *y) for x, y in zip(xs, ys)]
        else:
            pts = [(*x, y) for x, y in zip(xs, ys)]
        return simplex_volume_in_embedding(pts)


class TriangleLoss(LossFunction):
    @property
    def n_neighbors(self):
        return 1

    def __call__(self, domain, subdomain, codomain_bounds, data):
        assert isinstance(domain, ConvexHull)
        neighbors = domain.neighbors(subdomain, self.n_neighbors)
        if not neighbors:
            return 0
        neighbor_points = set.union(*(set(n) - set(subdomain) for n in neighbors))

        neighbor_points = [domain.triangulation.vertices[p] for p in neighbor_points]

        simplex = [domain.triangulation.vertices[p] for p in subdomain]

        z = data[simplex[0]]
        if isinstance(z, Iterable):
            s = [(*x, *data[x]) for x in simplex]
            n = [(*x, *data[x]) for x in neighbor_points]
        else:
            s = [(*x, data[x]) for x in simplex]
            n = [(*x, data[x]) for x in neighbor_points]

        return sum(simplex_volume_in_embedding([*s, neigh]) for neigh in n) / len(
            neighbors
        )


class CurvatureLoss(LossFunction):
    def __init__(self, exploration=0.05):
        self.exploration = exploration

    @property
    def n_neighbors(self):
        return 1

    def __call__(self, domain, subdomain, codomain_bounds, data):
        dim = domain.ndim

        loss_input_volume = domain.volume(subdomain)
        triangle_loss = TriangleLoss()

        loss_curvature = triangle_loss(domain, subdomain, codomain_bounds, data)
        return (
            loss_curvature + self.exploration * loss_input_volume ** ((2 + dim) / dim)
        ) ** (1 / (2 + dim))


class LearnerND(BaseLearner):
    def __init__(self, f, bounds, loss=None):

        if len(bounds) == 1:
            (a, b), = (boundary_points,) = bounds
            self.domain = Interval(a, b)
            self.loss_function = loss or DistanceLoss()
            self.ndim = 1
        else:
            boundary_points = sorted(tuple(p) for p in itertools.product(*bounds))
            self.domain = ConvexHull(scipy.spatial.ConvexHull(boundary_points))
            self.loss_function = loss or EmbeddedVolumeLoss()
            self.ndim = len(boundary_points[0])

        self.boundary_points = boundary_points
        self.queue = Queue()
        self.data = dict()  # Contains the evaluated data only
        self.pending_points = set()
        self.need_loss_update_factor = 1.1
        self.function = f
        self.n_asked = 0

        # As an optimization we keep a map from subdomain to loss.
        # This is updated in 'self.priority' whenever the loss function is evaluated
        # for a new subdomain. 'self.tell_many' removes subdomains from here when
        # they are split, and also removes neighboring subdomains from here (to force
        # a loss function recomputation)
        self.losses = dict()

        # We must wait until the boundary points have been evaluated before we can
        # set these attributes.
        self._initialized = False
        self.vdim = None
        self.codomain_bounds = None
        self.codomain_scale_at_last_update = None

        for subdomain in self.domain.subdomains():
            self.queue.insert(subdomain, priority=self.priority(subdomain))

    def _finalize_initialization(self):
        assert all(x in self.data for x in self.boundary_points)

        self._initialized = True

        vals = list(self.data.values())
        codomain_min = np.min(vals, axis=0)
        codomain_max = np.max(vals, axis=0)
        self.codomain_bounds = (codomain_min, codomain_max)
        self.codomain_scale_at_last_update = codomain_max - codomain_min

        try:
            self.vdim = len(np.squeeze(self.data[self.boundary_points[0]]))
        except TypeError:  # Trying to take the length of a number
            self.vdim = 1

        for x in self.data:
            if x in self.boundary_points:
                continue
            self.domain.split_at(x)

        self.queue = Queue()
        for subdomain in self.domain.subdomains():
            self.queue.insert(subdomain, priority=self.priority(subdomain))

    def priority(self, subdomain):
        if self._initialized:
            if subdomain in self.losses:
                L_0 = self.losses[subdomain]
            else:
                L_0 = self.loss_function(self.domain, subdomain, self.codomain_bounds, self.data)
                self.losses[subdomain] = L_0
        else:
            # Before we have all the boundary points we can't calculate losses because we
            # do not have enough data. We just assign a constant loss to each subdomain.
            L_0 = 1

        subvolumes = self.domain.subvolumes(subdomain)
        return (max(subvolumes) / sum(subvolumes)) * L_0

    def ask(self, n, tell_pending=True):
        if self.n_asked >= len(self.boundary_points):
            points, losses = self._ask(n, tell_pending)
        else:
            points = self.boundary_points[self.n_asked:self.n_asked + n]
            # The boundary points should always be evaluated with the highest priority
            losses = [float('inf')] * len(points)
            if tell_pending:
                for x in points:
                    self.pending_points.add(x)
            n_extra = n - len(points)
            if n_extra > 0:
                extra_points, extra_losses = self._ask(n_extra, tell_pending)
                points += tuple(extra_points)
                losses += tuple(extra_losses)

        if tell_pending:
            self.n_asked += n

        return points, losses

    def _ask(self, n, tell_pending):
        new_points = []
        point_priorities = []
        for _ in range(n):
            subdomain, _ = self.queue.peek()
            (new_point,), affected_subdomains = self.domain.insert_points(subdomain, 1)
            self.pending_points.add(new_point)
            for subdomain in affected_subdomains:
                self.queue.update(subdomain, priority=self.priority(subdomain))
            new_points.append(new_point)
            point_priorities.append(self.priority(subdomain))

        if not tell_pending:
            affected_subdomains = set()
            for point in new_points:
                self.pending_points.remove(point)
                affected_subdomains.update(self.domain.remove(point))
            for subdomain in affected_subdomains:
                self.queue.update(subdomain, priority=self.priority(subdomain))
        return new_points, point_priorities

    def tell_pending(self, x):
        self.pending_points.add(x)
        affected_subdomains = self.domain.insert(x)
        for subdomain in affected_subdomains:
            self.queue.update(subdomain, priority=self.priority(subdomain))

    def tell_many(self, xs, ys):
        for x, y in zip(xs, ys):
            self.data[x] = y

        if not self._initialized:
            if all(x in self.data for x in self.boundary_points):
                self._finalize_initialization()
            return

        need_loss_update = self._update_codomain_bounds(ys)

        old = set()
        new = set()
        for x in xs:
            old_subdomains, new_subdomains = self.domain.split_at(x)
            old.update(old_subdomains)
            new.update(new_subdomains)
        # remove any subdomains that were new at some point but are now old
        new -= old

        for subdomain in old:
            self.queue.remove(subdomain)
            del self.losses[subdomain]

        if need_loss_update:
            # Need to recalculate all priorities anyway
            self.queue = Queue(
                (subdomain, self.priority(subdomain))
                for subdomain in itertools.chain(self.queue.items(), new)
            )
        else:
            # Compute the priorities for the new subdomains and re-compute the
            # priorities for the neighboring subdomains, if necessary.
            for subdomain in new:
                self.queue.insert(subdomain, priority=self.priority(subdomain))

            if self.loss_function.n_neighbors > 0:
                subdomains_to_update = set()
                for subdomain in new:
                    subdomains_to_update.update(
                        self.domain.neighbors(subdomain, self.loss_function.n_neighbors)
                    )
                subdomains_to_update -= new
                for subdomain in subdomains_to_update:
                    del self.losses[subdomain]  # Force loss recomputation
                    self.queue.update(subdomain, priority=self.priority(subdomain))

    def _update_codomain_bounds(self, ys):
        mn, mx = self.codomain_bounds
        if self.vdim == 1:
            mn = min(mn, *ys)
            mx = max(mx, *ys)
        else:
            mn = np.min(np.vstack([mn, ys]), axis=0)
            mx = np.max(np.vstack([mx, ys]), axis=0)
        self.codomain_bounds = (mn, mx)

        scale = mx - mn

        scale_factor = scale / self.codomain_scale_at_last_update
        if self.vdim == 1:
            need_loss_update = scale_factor > self.need_loss_update_factor
        else:
            need_loss_update = np.any(scale_factor > self.need_loss_update_factor)
        if need_loss_update:
            self.codomain_scale_at_last_update = scale
            return True
        else:
            return False

    def remove_unfinished(self):
        self.pending_points = set()
        cleared_subdomains = self.domain.clear_subdomains()
        # Subdomains who had internal points removed need their priority updating
        for subdomain in cleared_subdomains:
            self.queue.update(subdomain, priority=self.priority(subdomain))

    def loss(self, real=True):
        if real:
            # NOTE: O(N) in the number of subintervals, but with a low prefactor.
            #       We have to do this because the queue is sorted in *priority*
            #       order, and it's possible that a subinterval with a high loss
            #       may have a low priority (if there are many pending points).
            return max(self.losses.values())
        else:
            # This depends on the implementation of 'self.priority'. Currently
            # it returns a tuple (priority, loss).
            _, priority = self.queue.peek()
            return priority

    def plot(self, **kwargs):
        if isinstance(self.domain, Interval):
            return self._plot_1d(**kwargs)
        elif isinstance(self.domain, ConvexHull):
            return self._plot_nd(**kwargs)

    def _plot_nd(self, n=None, tri_alpha=0):
        # XXX: Copied from LearnerND. At the moment we reach deep into internal
        #      datastructures of self.domain. We should see what data we need and
        #      add APIs to 'Domain' to support this.
        hv = ensure_holoviews()
        if self.vdim > 1:
            raise NotImplementedError(
                "holoviews currently does not support", "3D surface plots in bokeh."
            )
        if self.ndim != 2:
            raise NotImplementedError(
                "Only 2D plots are implemented: You can "
                "plot a 2D slice with 'plot_slice'."
            )
        x, y = self.domain.bounding_box
        lbrt = x[0], y[0], x[1], y[1]

        if len(self.data) >= 4:
            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                scale_factor = 1  # np.product(np.diag(self._transform))
                min_volume = min(map(self.domain.volume, self.domain.subdomains()))
                a_sq = np.sqrt(scale_factor * min_volume)
                n = max(10, int(0.658 / a_sq))

            xs = ys = np.linspace(0, 1, n)
            xs = xs * (x[1] - x[0]) + x[0]
            ys = ys * (y[1] - y[0]) + y[0]
            keys = np.array(list(self.data.keys()))
            values = np.array(list(self.data.values()))
            interpolator = scipy.interpolate.LinearNDInterpolator(keys, values)
            z = interpolator(xs[:, None], ys[None, :]).squeeze()

            im = hv.Image(np.rot90(z), bounds=lbrt)

            if tri_alpha:
                points = np.array(
                    [
                        [self.domain.triangulation.vertices[i] for i in s]
                        for s in self.domain.subdomains()
                    ]
                )
                points = np.pad(
                    points[:, [0, 1, 2, 0], :],
                    pad_width=((0, 0), (0, 1), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                ).reshape(-1, 2)
                tris = hv.EdgePaths([points])
            else:
                tris = hv.EdgePaths([])
        else:
            im = hv.Image([], bounds=lbrt)
            tris = hv.EdgePaths([])

        im_opts = dict(cmap="viridis")
        tri_opts = dict(line_width=0.5, alpha=tri_alpha)
        no_hover = dict(plot=dict(inspection_policy=None, tools=[]))

        return im.opts(style=im_opts) * tris.opts(style=tri_opts, **no_hover)

    def _plot_1d(self):
        assert isinstance(self.domain, Interval)
        hv = ensure_holoviews()

        xs, ys = zip(*sorted(self.data.items())) if self.data else ([], [])
        if self.vdim == 1:
            p = hv.Path([]) * hv.Scatter((xs, ys))
        else:
            p = hv.Path((xs, ys)) * hv.Scatter([])

        # Plot with 5% empty margins such that the boundary points are visible
        a, b = self.domain.bounds
        margin = 0.05 * (b - a)
        plot_bounds = (a - margin, b + margin)

        return p.redim(x=dict(range=plot_bounds))

    def _get_data(self):
        pass

    def _set_data(self, data):
        pass
