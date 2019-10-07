from math import sqrt
import itertools
from collections.abc import Iterable

import numpy as np
import scipy.spatial
import scipy.interpolate
from sortedcontainers import SortedList, SortedDict

from adaptive.learner.base_learner import BaseLearner
from adaptive.learner.triangulation import Triangulation, simplex_volume_in_embedding
from adaptive.notebook_integration import ensure_holoviews


class Domain:
    def insert_points(self, subdomain, n):
        """Insert 'n' points into 'subdomain'.

        May not return a point on the boundary of subdomain.
        """

    def insert_into(self, subdomain, x):
        """Insert 'x' into 'subdomain'.

        Raises
        ------
        ValueError : if x is outside subdomain or exists already
        NotImplementedError : if x is on the boundary of subdomain
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

    def which_subdomain(self, x):
        """Return the subdomain that contains 'x'.

        Raises
        ------
        ValueError : if x is outside of the domain
        NotImplementedError : if x is on a subdomain boundary
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

    def insert_points(self, subdomain, n, *, _check_membership=True):
        assert n > 0
        if _check_membership and subdomain not in self:
            raise ValueError("{} is not present in this interval".format(subdomain))
        try:
            p = self.sub_intervals[subdomain]
        except KeyError:  # No points yet in the interior of this subdomain
            a, b = subdomain
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

        return points

    def insert_into(self, subdomain, x, *, _check_membership=True):
        a, b = subdomain
        if _check_membership:
            if subdomain not in self:
                raise ValueError("{} is not present in this interval".format(subdomain))
            if not (a < x < b):
                raise ValueError("{} is not in ({}, {})".format(x, a, b))

        try:
            p = self.sub_intervals[subdomain]
        except KeyError:
            self.sub_intervals[subdomain] = SortedList([a, x, b])
        else:
            p.add(x)

    def split_at(self, x, *, _check_membership=True):
        a, b = self.bounds
        if _check_membership:
            if not (a < x < b):
                raise ValueError("Can only split at points within the interval")
            if x in self.points:
                raise ValueError("Cannot split at an existing point")

        p = self.points
        i = p.bisect_left(x)
        a, b = old_interval = p[i - 1], p[i]
        new_intervals = [(a, x), (x, b)]

        p.add(x)
        try:
            sub_points = self.sub_intervals.pop(old_interval)
        except KeyError:
            pass
        else:  # update sub_intervals
            for ival in new_intervals:
                new_sub_points = SortedList(sub_points.irange(*ival))
                if x not in new_sub_points:
                    new_sub_points.add(x)
                if len(new_sub_points) > 2:
                    self.sub_intervals[ival] = new_sub_points

        return [old_interval], new_intervals

    def which_subdomain(self, x):
        a, b = self.bounds
        if not (a <= x <= b):
            raise ValueError("{} is outside the interval".format(x))
        p = self.points
        i = p.bisect_left(x)
        if p[i] == x:
            raise NotImplementedError("{} is on a subdomain boundary".format(x))
        return (p[i], p[i + 1])

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
        # interior of the subdomain.
        self.sub_domains = dict()

    @property
    def bounding_box(self):
        hull_points = self.bounds.points[self.bounds.vertices]
        return tuple(zip(hull_points.min(axis=0), hull_points.max(axis=0)))

    def insert_points(self, subdomain, n, *, _check_membership=True):
        assert n > 0
        tri = self.triangulation
        if _check_membership and subdomain not in tri.simplices:
            raise ValueError("{} is not present in this domain".format(subdomain))
        try:
            subtri = self.sub_domains[subdomain]
        except KeyError:  # No points in the interior of this subdomain yet
            subtri = Triangulation([tri.vertices[x] for x in subdomain])
            self.sub_domains[subdomain] = subtri

        # Choose new points in the centre of the largest sub-subdomain
        # of this subdomain
        points = []
        for _ in range(n):
            # O(N) in the number of sub-simplices, but typically we only have a few
            largest_simplex = max(subtri.simplices, key=lambda s: subtri.volume(s))
            simplex_vertices = np.array([subtri.vertices[s] for s in largest_simplex])
            # XXX: choose the centre of the simplex. Update this to be able to handle
            #      choosing points on a face. This requires updating the ABC and having
            #      this method also return the subdomains that are affected by the new
            #      points
            point = np.average(simplex_vertices, axis=0)
            subtri.add_point(point, largest_simplex)
            points.append(point)

        return [tuple(p) for p in points]

    def insert_into(self, subdomain, x, *, _check_membership=True):
        tri = self.triangulation
        if _check_membership:
            if subdomain not in tri.simplices:
                raise ValueError("{} is not present in this domain".format(subdomain))
            if not tri.point_in_simplex(x, subdomain):
                raise ValueError("{} is not present in this subdomain".format(x))

        try:
            subtri = self.sub_domains[subdomain]
        except KeyError:  # No points in the interior of this subdomain yet
            subtri = Triangulation([tri.vertices[x] for x in subdomain])
            self.sub_domains[subdomain] = subtri

        subtri.add_point(x)

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

        interior_points = []
        for d in old_subdomains:
            try:
                subtri = self.sub_domains.pop(d)
            except KeyError:
                continue
            else:
                # Get the points in the interior of the subtriangulation
                verts = set(range(len(subtri.vertices))) - subtri.hull
                assert verts
                verts = np.array([subtri.vertices[i] for i in verts])
                # Remove 'x' if it is one of the points
                verts = verts[np.all(verts != x, axis=1)]
                interior_points.append(verts)
        if interior_points:
            interior_points = np.vstack(interior_points)
        for p in interior_points:
            # XXX: handle case where points lie on simplex boundaries
            for subdomain in new_subdomains:
                if tri.point_in_simplex(p, subdomain):
                    try:
                        subtri = self.sub_domains[subdomain]
                    except KeyError:  # No points in this subdomain yet
                        subtri = Triangulation([tri.vertices[i] for i in subdomain])
                        self.sub_domains[subdomain] = subtri
                    subtri.add_point(p)
                    break
            else:
                assert False, "{} was not in the interior of new simplices".format(x)

        return old_subdomains, new_subdomains

    def which_subdomain(self, x):
        tri = self.triangulation
        member = np.array([tri.point_in_simplex(x, s) for s in tri.simplices])
        n_simplices = member.sum()
        if n_simplices < 1:
            raise ValueError("{} is not in the domain".format(x))
        elif n_simplices == 1:
            return member.argmax()
        else:
            raise ValueError("{} is on a subdomain boundary".format(x))

    def transform(self, x):
        # XXX: implement this
        raise NotImplementedError()

    def neighbors(self, subdomain, n=1):
        "Return all neighboring subdomains up to degree 'n'."
        tri = self.triangulation
        neighbors = {subdomain}
        for _ in range(n):
            for face in tri.faces(simplices=neighbors):
                neighbors.update(tri.containing(face))
        neighbors.remove(subdomain)
        return neighbors

    def subdomains(self):
        "Return all the subdomains in the domain."
        return self.triangulation.simplices

    def clear_subdomains(self):
        """Remove all points from the interior of subdomains.

        Returns
        -------
        subdomains : the subdomains who's interior points were removed
        """
        sub_domains = list(self.sub_domains.keys())
        self.sub_domains = dict()
        return sub_domains

    def volume(self, subdomain):
        "Return the volume of a subdomain."
        return self.triangulation.volume(subdomain)

    def subvolumes(self, subdomain):
        "Return the volumes of the sub-subdomains."
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
        "Return an iterator over the items in the queue in priority order."
        return reversed(self._queue.values())

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


def _scaled_loss(loss, domain, subdomain, codomain_bounds, data):
    subvolumes = domain.subvolumes(subdomain)
    max_relative_subvolume = max(subvolumes) / sum(subvolumes)
    L_0 = loss(domain, subdomain, codomain_bounds, data)
    return max_relative_subvolume * L_0


class LearnerND(BaseLearner):
    def __init__(self, f, bounds, loss=None):

        if len(bounds) == 1:
            (a, b), = (bound_points,) = bounds
            self.domain = Interval(a, b)
            self.loss = loss or DistanceLoss()
            self.ndim = 1
        else:
            bound_points = sorted(tuple(p) for p in itertools.product(*bounds))
            self.domain = ConvexHull(scipy.spatial.ConvexHull(bound_points))
            self.loss = loss or EmbeddedVolumeLoss()
            self.ndim = len(bound_points[0])

        self.queue = Queue()
        self.data = dict()
        self.function = f

        # Evaluate boundary points right away to avoid handling edge
        # cases in the ask and tell logic
        for x in bound_points:
            self.data[x] = f(x)

        vals = list(self.data.values())
        codomain_min = np.min(vals, axis=0)
        codomain_max = np.max(vals, axis=0)
        self.codomain_bounds = (codomain_min, codomain_max)
        self.codomain_scale_at_last_update = codomain_max - codomain_min

        self.need_loss_update_factor = 1.1

        try:
            self.vdim = len(np.squeeze(self.data[x]))
        except TypeError:  # Trying to take the length of a number
            self.vdim = 1

        for subdomain in self.domain.subdomains():
            # NOTE: could just call 'self.loss' here, as we *know* that each
            #       subdomain does not have internal points.
            loss = _scaled_loss(
                self.loss, self.domain, subdomain, self.codomain_bounds, self.data
            )
            self.queue.insert(subdomain, priority=loss)

    def ask(self, n, tell_pending=True):
        if not tell_pending:
            # XXX: handle this case
            raise RuntimeError("tell_pending=False not supported yet")
        new_points = []
        new_losses = []
        for _ in range(n):
            subdomain, _ = self.queue.pop()
            new_point, = self.domain.insert_points(subdomain, 1)
            self.data[new_point] = None
            new_loss = _scaled_loss(
                self.loss, self.domain, subdomain, self.codomain_bounds, self.data
            )
            self.queue.insert(subdomain, priority=new_loss)
            new_points.append(new_point)
            new_losses.append(new_loss)
        return new_points, new_losses

    def tell_pending(self, x):
        self.data[x] = None
        subdomain = self.domain.which_subdomain(x)
        self.domain.insert_into(subdomain, x)
        loss = _scaled_loss(
            self.loss, self.domain, subdomain, self.codomain_bounds, self.data
        )
        self.queue.update(subdomain, priority=loss)

    def tell_many(self, xs, ys):
        for x, y in zip(xs, ys):
            self.data[x] = y

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

        if need_loss_update:
            # Need to recalculate all losses anyway
            self.queue = Queue(
                (
                    subdomain,
                    _scaled_loss(
                        self.loss,
                        self.domain,
                        subdomain,
                        self.codomain_bounds,
                        self.data,
                    ),
                )
                for subdomain in itertools.chain(self.queue.items(), new)
            )
        else:
            # Compute the losses for the new subdomains and re-compute the
            # losses for the neighboring subdomains, if necessary.
            for subdomain in new:
                loss = _scaled_loss(
                    self.loss, self.domain, subdomain, self.codomain_bounds, self.data
                )
                self.queue.insert(subdomain, priority=loss)

            if self.loss.n_neighbors > 0:
                subdomains_to_update = sum(
                    (set(self.domain.neighbors(d, self.loss.n_neighbors)) for d in new),
                    set(),
                )
                subdomains_to_update -= new
                for subdomain in subdomains_to_update:
                    loss = _scaled_loss(
                        self.loss,
                        self.domain,
                        subdomain,
                        self.codomain_bounds,
                        self.data,
                    )
                    self.queue.update(subdomain, priority=loss)

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
        self.data = {k: v for k, v in self.data.items() if v is not None}
        cleared_subdomains = self.domain.clear_subdomains()
        # Subdomains who had internal points removed need their losses updating
        for subdomain in cleared_subdomains:
            loss = _scaled_loss(
                self.loss, self.domain, subdomain, self.codomain_bounds, self.data
            )
            self.queue.update(subdomain, priority=loss)

    def loss(self):
        _, loss = self.queue.peek()
        return loss

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
