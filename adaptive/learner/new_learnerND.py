from math import sqrt
import itertools

import numpy as np
import sortedcontainers

from adaptive.learner.base_learner import BaseLearner
from adaptive.notebook_integration import ensure_holoviews


class Domain:
    def insert_points(self, subdomain, n):
        "Insert 'n' points into 'subdomain'."

    def insert_into(self, subdomain, x):
        """Insert 'x' into 'subdomain'.

        Raises
        ------
        ValueError : if x is outside of subdomain or on its boundary
        """

    def split_at(self, x):
        """Split the domain at 'x'.

        Returns
        -------
        old_subdomains : list of subdomains
            The subdomains that were removed when splitting at 'x'.
        new_subdomains : list of subdomains
            The subdomains that were added when splitting at 'x'.
        """

    def which_subdomain(self, x):
        """Return the subdomain that contains 'x'.

        Raises
        ------
        ValueError: if x is on a subdomain boundary
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
        self.points = sortedcontainers.SortedList([a, b])

    def insert_points(self, subdomain, n, *, _check_membership=True):
        if _check_membership and subdomain not in self:
            raise ValueError("{} is not present in this interval".format(subdomain))
        try:
            p = self.sub_intervals[subdomain]
        except KeyError:  # first point in the interior of this subdomain
            a, b = subdomain
            points = np.linspace(a, b, 2 + n)
            self.sub_intervals[subdomain] = sortedcontainers.SortedList(points)
            return points[1:-1]

        # XXX: allow this
        if n != 1:
            raise ValueError("Can't add more than one point to a full subinterval")

        subsubdomains = zip(p, p.islice(1))
        a, b = max(subsubdomains, key=lambda ival: ival[1] - ival[0])
        m = a + (b - a) / 2
        p.add(m)
        return [m]

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
            self.sub_intervals[subdomain] = sortedcontainers.SortedList([a, x, b])
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
                new_sub_points = sortedcontainers.SortedList(sub_points.irange(*ival))
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
            raise ValueError("{} belongs to 2 subdomains".format(x))
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
        "Return all neighboring subdomains up to degree 'n'."
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

    def points(self, subdomain):
        "Return all the points that define a given subdomain."
        try:
            return self.sub_intervals[subdomain]
        except KeyError:
            return subdomain

    def subdomains(self):
        "Return all the subdomains in the domain."
        p = self.points
        return zip(p, p.islice(1))

    def clear_subdomains(self):
        subdomains = list(self.sub_intervals.keys())
        self.sub_intervals = dict()
        return subdomains

    def volume(self, subdomain):
        "Return the volume of a subdomain"
        a, b = subdomain
        return b - a

    def subvolumes(self, subdomain):
        "Return the volumes of the sub-subdomains."
        try:
            p = self.sub_intervals[subdomain]
        except KeyError:
            return [self.volume(subdomain)]
        else:
            return [self.volume(s) for s in zip(p, p.islice(1))]


class Queue:
    """Priority queue supporting update and removal at arbitrary position.

    Parameters
    ----------
    entries : iterable of (item, priority)
        The initial data in the queue. Providing this is faster than
        calling 'insert' a bunch of times.
    """

    def __init__(self, entries=()):
        self._queue = sortedcontainers.SortedDict(
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
        self._items = sortedcontainers.SortedList(
            ((v, k) for k, v in self._queue.items())
        )

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

    def update(self, item, new_priority):
        """Update 'item' in the queue with the given priority.

        Raises
        ------
        KeyError : if 'item' is not in the queue.
        """
        i = self._items.bisect_left((item, ()))
        should_be, key = self._items[i]
        if item != should_be:
            raise KeyError("item is not in queue")

        _, n = key
        new_key = (new_priority, n)

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
        # XXX: this is specialised to 1D
        a, b = subdomain
        ya, yb = data[a], data[b]
        return sqrt((b - a) ** 2 + (yb - ya) ** 2)


def _scaled_loss(loss, domain, subdomain, codomain_bounds, data):
    subvolumes = domain.subvolumes(subdomain)
    max_relative_subvolume = max(subvolumes) / sum(subvolumes)
    L_0 = loss(domain, subdomain, codomain_bounds, data)
    return max_relative_subvolume * L_0


class LearnerND(BaseLearner):
    def __init__(self, f, bounds, loss=None):

        if len(bounds) == 1:
            (a, b), = bound_points, = bounds
            self.domain = Interval(a, b)
            self.loss = loss or DistanceLoss()
        else:
            raise ValueError("Can only handle 1D functions for now")

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

        d, = self.domain.subdomains()
        loss = self.loss(self.domain, d, self.codomain_bounds, self.data)
        self.queue.insert(d, priority=loss)

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
            subdomains = itertools.chain(self.queue.items(), new)
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

    def plot(self):
        # XXX: specialized to 1D
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
