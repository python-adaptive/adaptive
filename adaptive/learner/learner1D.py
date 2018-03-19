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
        self.data_interp = {}

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self.neighbors = sortedcontainers.SortedDict()
        self.neighbors_combined = sortedcontainers.SortedDict()

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [list(bounds), [np.inf, -np.inf]]

        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [bounds[1] - bounds[0], 0]
        self._oldscale = deepcopy(self._scale)

        self.bounds = list(bounds)

        self._vdim = None

    @property
    def vdim(self):
        return 1 if self._vdim is None else self._vdim

    @property
    def data_combined(self):
        return {**self.data, **self.data_interp}

    @property
    def npoints(self):
        return len(self.data)

    def loss(self, real=True):
        losses = self.losses if real else self.losses_combined
        if len(losses) == 0:
            return float('inf')
        else:
            return max(losses.values())

    def update_losses(self, x, data, neighbors, losses):
        x_lower, x_upper = neighbors[x]
        if x_lower is not None:
            losses[x_lower, x] = self.loss_per_interval((x_lower, x),
                                                        self._scale, data)
        if x_upper is not None:
            losses[x, x_upper] = self.loss_per_interval((x, x_upper),
                                                        self._scale, data)
        try:
            del losses[x_lower, x_upper]
        except KeyError:
            pass

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

    def add_point(self, x, y):
        real = y is not None

        if real:
            # Add point to the real data dict and pop from the unfinished
            # data_interp dict.
            self.data[x] = y
            self.data_interp.pop(x, None)

            if self._vdim is None:
                try:
                    self._vdim = len(np.squeeze(y))
                except TypeError:
                    self._vdim = 1

            # Invalidate interpolated neighbors of new point
            i = self.data.bisect_left(x)
            if i == 0:
                x_left = self.data.iloc[0]
                for _x in self.data_interp:
                    if _x < x_left:
                        self.data_interp[_x] = None
            elif i == len(self.data):
                x_right = self.data.iloc[-1]
                for _x in self.data_interp:
                    if _x > x_right:
                        self.data_interp[_x] = None
            else:
                x_left, x_right = self.data.iloc[i-1], self.data.iloc[i]
                for _x in self.data_interp:
                    if x_left < _x < x_right:
                        self.data_interp[_x] = None

        else:
            # The keys of data_interp are the unknown points
            self.data_interp[x] = None

        # Update the neighbors
        self.update_neighbors(x, self.neighbors_combined)
        if real:
            self.update_neighbors(x, self.neighbors)

        # Update the scale
        self.update_scale(x, y)

        # Interpolate
        for _x, _y in self.data_interp.items():
            if _y is None:
                if len(self.data) >=2:
                    i = self.data.bisect_left(_x)
                    if i == 0:
                        i_left, i_right = (0, 1)
                    elif i == len(self.data):
                        i_left, i_right = (-2, -1)
                    else:
                        i_left, i_right = (i - 1, i)
                    x_left, x_right = self.data.iloc[i_left], self.data.iloc[i_right]
                    y_left, y_right = self.data[x_left], self.data[x_right]
                    dx = x_right - x_left
                    dy = y_right - y_left
                    self.data_interp[_x] = (dy / dx) * (_x - x_left) + y_left

        # Update the losses
        self.update_losses(x, self.data_combined, self.neighbors_combined,
                           self.losses_combined)
        if real:
            self.update_losses(x, self.data, self.neighbors, self.losses)

        # If the scale has doubled, recompute all losses.
        if self._scale > self._oldscale * 2:
            self.losses = {xs: self.loss_per_interval(xs, self._scale, self.data)
                           for xs in self.losses}
            self.losses_combined = {x: self.loss_per_interval(x, self._scale,
                                                              self.data_combined)
                                    for x in self.losses_combined}
            self._oldscale = self._scale

    def choose_points(self, n, add_data=True):
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
            if bound not in self.data and bound not in self.data_interp:
                points.append(bound)

        # Ensure we return exactly 'n' points.
        if points:
            loss_improvements = [float('inf')] * n
            if n <= 2:
                points = points[:n]
            else:
                points = np.linspace(*self.bounds, n)
        else:
            def xs(x, n):
                if n == 1:
                    return []
                else:
                    step = (x[1] - x[0]) / n
                    return [x[0] + step * i for i in range(1, n)]

            # Calculate how many points belong to each interval.
            quals = [(-loss, x_range, 1) for (x_range, loss) in
                     self.losses_combined.items()]

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
            self.add_data(points, itertools.repeat(None))

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
        self.data_interp = {}
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)
