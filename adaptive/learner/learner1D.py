# -*- coding: utf-8 -*-
from copy import deepcopy
import heapq
import itertools
import math

import holoviews as hv
import numpy as np
import sortedcontainers
import scipy.interpolate

from .base_learner import BaseLearner

class Learner1D(BaseLearner):
    """Learns and predicts a function 'f:ℝ → ℝ'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single real parameter and
        return a real number.
    bounds : pair of reals
        The bounds of the interval on which to learn 'function'.
    """

    def __init__(self, function, bounds, vector_output=False):
        self.function = function

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

        self.vector_output = vector_output

    @property
    def data_combined(self):
        return {**self.data, **self.data_interp}

    def interval_loss(self, x_left, x_right, data):
        """Calculate loss in the interval x_left, x_right.

        Currently returns the rescaled length of the interval. If one of the
        y-values is missing, returns 0 (so the intervals with missing data are
        never touched. This behavior should be improved later.
        """
        y_right, y_left = data[x_right], data[x_left]
        x_scale, y_scale = self._scale
        dx = (x_right - x_left) / x_scale
        if y_scale == 0:
            loss = dx
        else:
            dy = (y_right - y_left) / y_scale
            if self.vector_output:
                loss = np.hypot(dx, dy).max()
            else:
                loss = math.hypot(dx, dy)
        return loss

    def loss(self, real=True):
        losses = self.losses if real else self.losses_combined
        if len(losses) == 0:
            return float('inf')
        else:
            return max(losses.values())

    def update_losses(self, x, data, neighbors, losses):
        x_lower, x_upper = neighbors[x]
        if x_lower is not None:
            losses[x_lower, x] = self.interval_loss(x_lower, x, data)
        if x_upper is not None:
            losses[x, x_upper] = self.interval_loss(x, x_upper, data)
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
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        if y is not None:
            y1, y2 = (min(y), max(y)) if self.vector_output else (y, y)
            self._bbox[1][0] = min(self._bbox[1][0], y1)
            self._bbox[1][1] = max(self._bbox[1][1], y2)

        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

    def add_point(self, x, y):
        real = y is not None

        if real:
            # Add point to the real data dict and pop from the unfinished
            # data_interp dict.
            self.data[x] = y
            try:
                del self.data_interp[x]
            except KeyError:
                pass
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
        if not real:
            self.data_interp = self.interpolate()

        # Update the losses
        self.update_losses(x, self.data_combined, self.neighbors_combined,
                           self.losses_combined)
        if real:
            self.update_losses(x, self.data, self.neighbors, self.losses)

        # If the scale has doubled, recompute all losses.
        if self._scale > self._oldscale * 2:
            self.losses = {xs: self.interval_loss(*xs, self.data)
                           for xs in self.losses}
            self.losses_combined = {x: self.interval_loss(*x,
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

    def interpolate(self, extra_points=None):
        xs = list(self.data.keys())
        ys = list(self.data.values())
        xs_unfinished = list(self.data_interp.keys())

        if extra_points is not None:
            xs_unfinished += extra_points

        if len(xs) < 2:
            interp_ys = np.zeros(len(xs_unfinished))
            if self.vector_output:
                interp_ys = interp_ys.reshape(-1, 1)
        else:
            if self.vector_output:
                ip = scipy.interpolate.interp1d(xs, np.transpose(ys),
                                                assume_sorted=True,
                                                bounds_error=False,
                                                fill_value=0)
                interp_ys = ip(xs_unfinished).T
            else:
                interp_ys = np.interp(xs_unfinished, xs, ys)

        data_interp = {x: y for x, y in zip(xs_unfinished, interp_ys)}

        return data_interp

    def plot(self):
        if not self.vector_output:
            if self.data:
                return hv.Scatter(self.data)
            else:
                return hv.Scatter([])
        else:
            if self.data:
                xs = list(self.data.keys())
                ys = np.array(list(self.data.values())).T
                return hv.Overlay([hv.Scatter((xs, y)) for y in ys])
            else:
                return hv.Overlay([hv.Scatter([])])

    def remove_unfinished(self):
        self.data_interp = {}
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)
