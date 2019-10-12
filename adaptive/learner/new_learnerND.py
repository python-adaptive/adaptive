import abc
import itertools
from collections.abc import Iterable
import math

import numpy as np
import scipy.interpolate

from adaptive.learner.base_learner import BaseLearner
from adaptive.learner.triangulation import simplex_volume_in_embedding
from adaptive.notebook_integration import ensure_holoviews
from adaptive.priority_queue import Queue
from adaptive.domain import Interval, ConvexHull


class LossFunction(metaclass=abc.ABCMeta):
    @abc.abtractproperty
    def n_neighbors(self):
        "The maximum degree of neighboring subdomains required."

    @abc.abstractmethod
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
        return math.sqrt((b - a) ** 2 + (yb - ya) ** 2)


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
            self.domain = ConvexHull(boundary_points)
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
                L_0 = self.loss_function(
                    self.domain, subdomain, self.codomain_bounds, self.data
                )
                self.losses[subdomain] = L_0
        else:
            # Before we have all the boundary points we can't calculate losses because we
            # do not have enough data. We just assign the subdomain volume as the loss.
            L_0 = self.domain.volume(subdomain)

        subvolumes = self.domain.subvolumes(subdomain)
        return (max(subvolumes) / sum(subvolumes)) * L_0

    def ask(self, n, tell_pending=True):
        if self.n_asked >= len(self.boundary_points):
            points, losses = self._ask(n, tell_pending)
        else:
            points = self.boundary_points[self.n_asked : self.n_asked + n]
            # The boundary points should always be evaluated with the highest priority
            losses = [math.inf] * len(points)
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
            # TODO: don't call 'priority' again here: we already called it above, we just
            #       need to identify 'subdomin' within 'affected_subdomains'. Maybe change
            #       the API of 'Domain.insert_points' to not return 'subdomain'...
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
        self.data.update(zip(xs, ys))
        self.pending_points -= set(xs)

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
