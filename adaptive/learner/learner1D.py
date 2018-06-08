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

        self.data = sortedcontainers.SortedDict()
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
        return 1 if self._vdim is None else self._vdim

    @property
    def npoints(self):
        return len(self.data)

    def loss(self, real=True):
        losses = self.losses if real else self.losses_combined
        if len(losses) == 0:
            return float('inf')
        else:
            return max(losses.values())

    def update_interpolated_losses_in_interval(self, x_lower, x_upper):
        if x_lower is not None and x_upper is not None:
            dx = x_upper - x_lower
            loss = self.loss_per_interval((x_lower, x_upper), self._scale, self.data)
            self.losses[x_lower, x_upper] = loss if abs(dx) > self._dx_eps else 0

            start = self.neighbors_combined.bisect_right(x_lower)
            end = self.neighbors_combined.bisect_left(x_upper)
            for i in range(start, end):
                a, b = self.neighbors_combined.iloc[i], self.neighbors_combined.iloc[i + 1]
                self.losses_combined[a, b] = (b - a) * self.losses[x_lower, x_upper] / dx
            if start == end:
                self.losses_combined[x_lower, x_upper] = self.losses[x_lower, x_upper]

    def update_losses(self, x, real=True):
        if real:
            x_lower, x_upper = self.get_neighbors(x, self.neighbors)
            self.update_interpolated_losses_in_interval(x_lower, x)
            self.update_interpolated_losses_in_interval(x, x_upper)
            self.losses.pop((x_lower, x_upper), None)
        else:
            losses_combined = self.losses_combined
            x_lower, x_upper = self.get_neighbors(x, self.neighbors)
            a, b = self.get_neighbors(x, self.neighbors_combined)
            if x_lower is not None and x_upper is not None:
                dx = x_upper - x_lower
                loss = self.losses[x_lower, x_upper]
                losses_combined[a, x] = ((x - a) * loss / dx
                                         if abs(x - a) > self._dx_eps else 0)
                losses_combined[x, b] = ((b - x) * loss  / dx
                                         if abs(b - x) > self._dx_eps else 0)
            else:
                if a is not None:
                    losses_combined[a, x] = float('inf')
                if b is not None:
                    losses_combined[x, b] = float('inf')

            losses_combined.pop((a, b), None)

    def get_neighbors(self, x, neighbors):
        if x in neighbors:
            return neighbors[x]
        return self.find_neighbors(x, neighbors)

    def find_neighbors(self, x, neighbors):
        pos = neighbors.bisect_left(x)
        x_lower = neighbors.iloc[pos-1] if pos != 0 else None
        x_upper = neighbors.iloc[pos] if pos != len(neighbors) else None
        return x_lower, x_upper

    def update_neighbors(self, x, neighbors):
        if x not in neighbors:  # The point is new
            x_lower, x_upper = self.find_neighbors(x, neighbors)
            neighbors[x] = [x_lower, x_upper]
            neighbors.get(x_lower, [None, None])[1] = x
            neighbors.get(x_upper, [None, None])[0] = x

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

    def _tell(self, x, y):
        real = y is not None

        if real:
            # Add point to the real data dict
            self.data[x] = y
            # remove from set of pending points
            self.pending_points.discard(x)

            if self._vdim is None:
                try:
                    self._vdim = len(np.squeeze(y))
                except TypeError:
                    self._vdim = 1
        else:
            # The keys of pending_points are the unknown points
            self.pending_points.add(x)

        # Update the neighbors
        self.update_neighbors(x, self.neighbors_combined)
        if real:
            self.update_neighbors(x, self.neighbors)

        # Update the scale
        self.update_scale(x, y)

        # Update the losses
        self.update_losses(x, real)

        # If the scale has increased enough, recompute all losses.
        if self._scale[1] > self._oldscale[1] * 2:

            for interval in self.losses:
                self.update_interpolated_losses_in_interval(*interval)

            self._oldscale = deepcopy(self._scale)


    def ask(self, n, add_data=True):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        if n == 0:
            return [], []

        # If the bounds have not been chosen yet, we choose them first.
        points = []
        for bound in self.bounds:
            if bound not in self.data and bound not in self.pending_points:
                points.append(bound)

        if len(points) == 2:
            # First time
            loss_improvements = [np.inf] * n
            points = np.linspace(*self.bounds, n)
        elif len(points) == 1:
            # Second time, if we previously returned just self.bounds[0]
            loss_improvements = [np.inf] * n
            points = np.linspace(*self.bounds, n + 1)[1:]
        else:
            def xs(x, n):
                if n == 1:
                    return []
                else:
                    step = (x[1] - x[0]) / n
                    return [x[0] + step * i for i in range(1, n)]

            # Calculate how many points belong to each interval.
            x_scale = self._scale[0]
            quals = [(-loss if not math.isinf(loss) else (x0 - x1) / x_scale, (x0, x1), 1)
                     for ((x0, x1), loss) in self.losses_combined.items()]

            heapq.heapify(quals)

            for point_number in range(n):
                quality, x, n = quals[0]
                heapq.heapreplace(quals, (quality * n / (n + 1), x, n + 1))

            points = list(itertools.chain.from_iterable(xs(x, n)
                          for quality, x, n in quals))

            loss_improvements = list(itertools.chain.from_iterable(
                                     itertools.repeat(-quality, n-1)
                                     for quality, x, n in quals))

        if add_data:
            self.tell(points, itertools.repeat(None))

        return points, loss_improvements

    def plot(self):
        hv = ensure_holoviews()
        if not self.data:
            p = hv.Scatter([]) * hv.Path([])
        elif not self.vdim > 1:
            p = hv.Scatter(self.data) * hv.Path([])
        else:
            xs = list(self.data.keys())
            ys = list(self.data.values())
            p = hv.Path((xs, ys)) * hv.Scatter([])

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (self.bounds[1] - self.bounds[0])
        plot_bounds = (self.bounds[0] - margin, self.bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))


    def remove_unfinished(self):
        self.pending_points = set()
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)
