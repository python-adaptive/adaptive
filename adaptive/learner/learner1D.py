# -*- coding: utf-8 -*-
from copy import deepcopy
import heapq
import itertools
import math

import numpy as np
import sortedcontainers

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner


def uniform_loss(interval, scale, function_values):
    """Loss function that samples the domain uniformly.

    Works with `~adaptive.Learner1D` only.

    Examples
    --------
    >>> def f(x):
    ...     return x**2
    >>>
    >>> learner = adaptive.Learner1D(f,
    ...                              bounds=(-1, 1),
    ...                              loss_per_interval=uniform_sampling_1d)
    >>>
    """
    x_left, x_right = interval
    x_scale, _ = scale
    dx = (x_right - x_left) / x_scale
    return dx


def default_loss(interval, scale, function_values):
    """Calculate loss on a single interval

    Currently returns the rescaled length of the interval. If one of the
    y-values is missing, returns 0 (so the intervals with missing data are
    never touched. This behavior should be improved later.
    """
    x_left, x_right = interval
    y_right, y_left = function_values[x_right], function_values[x_left]
    x_scale, y_scale = scale
    dx = (x_right - x_left) / x_scale
    if y_scale == 0:
        loss = dx
    else:
        dy = (y_right - y_left) / y_scale
        try:
            _ = len(dy)
            loss = np.hypot(dx, dy).max()
        except TypeError:
            loss = math.hypot(dx, dy)
    return loss


def linspace(x_left, x_right, n):
    """This is equivalent to
    'np.linspace(x_left, x_right, n, endpoint=False)[1:]',
    but it is 15-30 times faster for small 'n'."""
    if n == 1:
        # This is just an optimization
        return []
    else:
        step = (x_right - x_left) / n
        return [x_left + step * i for i in range(1, n)]


def _get_neighbors_from_list(xs):
    xs = np.sort(xs)
    xs_left = np.roll(xs, 1).tolist()
    xs_right = np.roll(xs, -1).tolist()
    xs_left[0] = None
    xs_right[-1] = None
    neighbors = {x: [x_L, x_R] for x, x_L, x_R
                 in zip(xs, xs_left, xs_right)}
    return sortedcontainers.SortedDict(neighbors)


class Learner1D(BaseLearner):
    """Learns and predicts a function 'f:ℝ → ℝ^N'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single real parameter and
        return a real number.
    bounds : pair of reals
        The bounds of the interval on which to learn 'function'.
    loss_per_interval: callable, optional
        A function that returns the loss for a single interval of the domain.
        If not provided, then a default is used, which uses the scaled distance
        in the x-y plane as the loss. See the notes for more details.

    Notes
    -----
    'loss_per_interval' takes 3 parameters: interval, scale, and function_values,
    and returns a scalar; the loss over the interval.

    interval : (float, float)
        The bounds of the interval.
    scale : (float, float)
        The x and y scale over all the intervals, useful for rescaling the
        interval loss.
    function_values : dict(float -> float)
        A map containing evaluated function values. It is guaranteed
        to have values for both of the points in 'interval'.
    """

    def __init__(self, function, bounds, loss_per_interval=None):
        self.function = function
        self.loss_per_interval = loss_per_interval or default_loss

        # A dict storing the loss function for each interval x_n.
        self.losses = {}
        self.losses_combined = {}

        self.data = {}
        self.pending_points = set()

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self.neighbors = sortedcontainers.SortedDict()
        self.neighbors_combined = sortedcontainers.SortedDict()

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [list(bounds), [np.inf, -np.inf]]

        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [bounds[1] - bounds[0], 0]
        self._oldscale = deepcopy(self._scale)

        # The precision in 'x' below which we set losses to 0.
        self._dx_eps = 2 * max(np.abs(bounds)) * np.finfo(float).eps

        self.bounds = list(bounds)

        self._vdim = None

    @property
    def vdim(self):
        if self._vdim is None:
            if self.data:
                y = next(iter(self.data.values()))
                try:
                    self._vdim = len(np.squeeze(y))
                except TypeError:
                    # Means we are taking the length of a float
                    self._vdim = 1
            else:
                return 1
        return self._vdim

    @property
    def npoints(self):
        return len(self.data)

    def loss(self, real=True):
        losses = self.losses if real else self.losses_combined
        return max(losses.values()) if len(losses) > 0 else float('inf')

    def update_interpolated_loss_in_interval(self, x_left, x_right):
        if x_left is not None and x_right is not None:
            dx = x_right - x_left
            if dx < self._dx_eps:
                loss = 0
            else:
                loss = self.loss_per_interval((x_left, x_right),
                                              self._scale, self.data)
            self.losses[x_left, x_right] = loss

            # Iterate over all interpolated intervals in between
            # x_left and x_right and set the newly interpolated loss.
            a, b = x_left, None
            while b != x_right:
                b = self.neighbors_combined[a][1]
                self.losses_combined[a, b] = (b - a) * loss / dx
                a = b

    def update_losses(self, x, real=True):
        # When we add a new point x, we should update the losses
        # (x_left, x_right) are the "real" neighbors of 'x'.
        x_left, x_right = self.find_neighbors(x, self.neighbors)
        # (a, b) are the neighbors of the combined interpolated
        # and "real" intervals.
        a, b = self.find_neighbors(x, self.neighbors_combined)

        # (a, b) is splitted into (a, x) and (x, b) so if (a, b) exists
        self.losses_combined.pop((a, b), None)  # we get rid of (a, b).

        if real:
            # We need to update all interpolated losses in the interval
            # (x_left, x) and (x, x_right). Since the addition of the point
            # 'x' could change their loss.
            self.update_interpolated_loss_in_interval(x_left, x)
            self.update_interpolated_loss_in_interval(x, x_right)

            # Since 'x' is in between (x_left, x_right),
            # we get rid of the interval.
            self.losses.pop((x_left, x_right), None)
            self.losses_combined.pop((x_left, x_right), None)
        elif x_left is not None and x_right is not None:
            # 'x' happens to be in between two real points,
            # so we can interpolate the losses.
            dx = x_right - x_left
            loss = self.losses[x_left, x_right]
            self.losses_combined[a, x] = (x - a) * loss / dx
            self.losses_combined[x, b] = (b - x) * loss / dx

        # (no real point left of x) or (no real point right of a)
        left_loss_is_unknown = ((x_left is None) or
                                (not real and x_right is None))
        if (a is not None) and left_loss_is_unknown:
            self.losses_combined[a, x] = float('inf')

        # (no real point right of x) or (no real point left of b)
        right_loss_is_unknown = ((x_right is None) or
                                 (not real and x_left is None))
        if (b is not None) and right_loss_is_unknown:
            self.losses_combined[x, b] = float('inf')

    @staticmethod
    def find_neighbors(x, neighbors):
        if x in neighbors:
            return neighbors[x]
        pos = neighbors.bisect_left(x)
        keys = neighbors.keys()
        x_left = keys[pos - 1] if pos != 0 else None
        x_right = keys[pos] if pos != len(neighbors) else None
        return x_left, x_right

    def update_neighbors(self, x, neighbors):
        if x not in neighbors:  # The point is new
            x_left, x_right = self.find_neighbors(x, neighbors)
            neighbors[x] = [x_left, x_right]
            neighbors.get(x_left, [None, None])[1] = x
            neighbors.get(x_right, [None, None])[0] = x

    def update_scale(self, x, y):
        """Update the scale with which the x and y-values are scaled.

        For a learner where the function returns a single scalar the scale
        is determined by the peak-to-peak value of the x and y-values.

        When the function returns a vector the learners y-scale is set by
        the level with the the largest peak-to-peak value.
        """
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        self._scale[0] = self._bbox[0][1] - self._bbox[0][0]
        if y is not None:
            if self.vdim > 1:
                try:
                    y_min = np.min([self._bbox[1][0], y], axis=0)
                    y_max = np.max([self._bbox[1][1], y], axis=0)
                except ValueError:
                    # Happens when `_bbox[1]` is a float and `y` a vector.
                    y_min = y_max = y
                self._bbox[1] = [y_min, y_max]
                self._scale[1] = np.max(y_max - y_min)
            else:
                self._bbox[1][0] = min(self._bbox[1][0], y)
                self._bbox[1][1] = max(self._bbox[1][1], y)
                self._scale[1] = self._bbox[1][1] - self._bbox[1][0]

    def tell(self, x, y):
        if x in self.data:
            # The point is already evaluated before
            return

        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        # Add point to the real data dict
        self.data[x] = y

        # remove from set of pending points
        self.pending_points.discard(x)

        if not self.bounds[0] <= x <= self.bounds[1]:
            return

        self.update_neighbors(x, self.neighbors_combined)
        self.update_neighbors(x, self.neighbors)
        self.update_scale(x, y)
        self.update_losses(x, real=True)

        # If the scale has increased enough, recompute all losses.
        if self._scale[1] > 2 * self._oldscale[1]:

            for interval in self.losses:
                self.update_interpolated_loss_in_interval(*interval)

            self._oldscale = deepcopy(self._scale)

    def tell_pending(self, x):
        if x in self.data:
            # The point is already evaluated before
            return
        self.pending_points.add(x)
        self.update_neighbors(x, self.neighbors_combined)
        self.update_losses(x, real=False)

    def tell_many(self, xs, ys, *, force=False):
        if not force and not (len(xs) > 0.5 * len(self.data) and len(xs) > 2):
            # Only run this more efficient method if there are
            # at least 2 points and the amount of points added are
            # at least half of the number of points already in 'data'.
            # These "magic numbers" are somewhat arbitrary.
            super().tell_many(xs, ys)
            return

        # Add data points
        self.data.update(zip(xs, ys))
        self.pending_points.difference_update(xs)

        # Get all data as numpy arrays
        points = np.array(list(self.data.keys()))
        values = np.array(list(self.data.values()))
        points_pending = np.array(list(self.pending_points))
        points_combined = np.hstack([points_pending, points])

        # Generate neighbors
        self.neighbors = _get_neighbors_from_list(points)
        self.neighbors_combined = _get_neighbors_from_list(points_combined)

        # Update scale
        self._bbox[0] = [points_combined.min(), points_combined.max()]
        self._bbox[1] = [values.min(axis=0), values.max(axis=0)]
        self._scale[0] = self._bbox[0][1] - self._bbox[0][0]
        self._scale[1] = np.max(self._bbox[1][1] - self._bbox[1][0])
        self._oldscale = deepcopy(self._scale)

        # Find the intervals for which the losses should be calculated.
        intervals, intervals_combined = [
            [(x_m, x_r) for x_m, (x_l, x_r) in neighbors.items()][:-1]
            for neighbors in (self.neighbors, self.neighbors_combined)]

        # The the losses for the "real" intervals.
        self.losses = {}
        for x_left, x_right in intervals:
            self.losses[x_left, x_right] = (
                self.loss_per_interval((x_left, x_right), self._scale, self.data)
                if x_right - x_left >= self._dx_eps else 0)

        # List with "real" intervals that have interpolated intervals inside
        to_interpolate = []

        self.losses_combined = {}
        for ival in intervals_combined:
            # If this interval exists in 'losses' then copy it otherwise
            # calculate it.
            if ival in self.losses:
                self.losses_combined[ival] = self.losses[ival]
            else:
                # Set all losses to inf now, later they might be udpdated if the
                # interval appears to be inside a real interval.
                self.losses_combined[ival] = np.inf
                x_left, x_right = ival
                a, b = to_interpolate[-1] if to_interpolate else (None, None)
                if b == x_left and (a, b) not in self.losses:
                    # join (a, b) and (x_left, x_right) --> (a, x_right)
                    to_interpolate[-1] = (a, x_right)
                else:
                    to_interpolate.append((x_left, x_right))

        for ival in to_interpolate:
            if ival in self.losses:
                # If this interval does not exist it should already
                # have an inf loss.
                self.update_interpolated_loss_in_interval(*ival)

    def ask(self, n, tell_pending=True):
        """Return n points that are expected to maximally reduce the loss."""
        points, loss_improvements = self._ask_points_without_adding(n)

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _ask_points_without_adding(self, n):
        """Return n points that are expected to maximally reduce the loss.
        Without altering the state of the learner"""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.
        # Return equally spaced points within each interval to which points
        # will be added.

        # XXX: when is this used and could we safely remove it without impacting performance?
        if n == 0:
            return [], []

        # If the bounds have not been chosen yet, we choose them first.
        missing_bounds = [b for b in self.bounds if b not in self.data
                          and b not in self.pending_points]

        if len(missing_bounds) >= n:
            return missing_bounds[:n], [np.inf] * n

        def finite_loss(loss, xs):
            # If the loss is infinite we return the
            # distance between the two points.
            return (loss if not math.isinf(loss)
                else (xs[1] - xs[0]) / self._scale[0])

        quals = [(-finite_loss(loss, x), x, 1)
                 for x, loss in self.losses_combined.items()]

        # Add bound intervals to quals if bounds were missing.
        if len(self.data) + len(self.pending_points) == 0:
            # We don't have any points, so return a linspace with 'n' points.
            return np.linspace(*self.bounds, n).tolist(), [np.inf] * n
        elif len(missing_bounds) > 0:
            # There is at least one point in between the bounds.
            all_points = list(self.data.keys()) + list(self.pending_points)
            intervals = [(self.bounds[0], min(all_points)),
                         (max(all_points), self.bounds[1])]
            for interval, bound in zip(intervals, self.bounds):
                if bound in missing_bounds:
                    qual = (-finite_loss(np.inf, interval), interval, 1)
                    quals.append(qual)

        # Calculate how many points belong to each interval.
        points, loss_improvements = self._subdivide_quals(
            quals, n - len(missing_bounds))

        points = missing_bounds + points
        loss_improvements = [np.inf] * len(missing_bounds) + loss_improvements

        return points, loss_improvements

    def _subdivide_quals(self, quals, n):
        # Calculate how many points belong to each interval.
        heapq.heapify(quals)

        for _ in range(n):
            quality, x, n = quals[0]
            if abs(x[1] - x[0]) / (n + 1) <= self._dx_eps:
                # The interval is too small and should not be subdivided.
                quality = np.inf
                # XXX: see https://gitlab.kwant-project.org/qt/adaptive/issues/104
            heapq.heapreplace(quals, (quality * n / (n + 1), x, n + 1))

        points = list(itertools.chain.from_iterable(
            linspace(*interval, n) for quality, interval, n in quals))

        loss_improvements = list(itertools.chain.from_iterable(
            itertools.repeat(-quality, n - 1)
            for quality, interval, n in quals))

        return points, loss_improvements

    def plot(self):
        hv = ensure_holoviews()
        if not self.data:
            p = hv.Scatter([]) * hv.Path([])
        elif not self.vdim > 1:
            p = hv.Scatter(self.data) * hv.Path([])
        else:
            xs, ys = zip(*sorted(self.data.items()))
            p = hv.Path((xs, ys)) * hv.Scatter([])

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (self.bounds[1] - self.bounds[0])
        plot_bounds = (self.bounds[0] - margin, self.bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))

    def remove_unfinished(self):
        self.pending_points = set()
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)
