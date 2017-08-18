# -*- coding: utf-8 -*-
import abc
import heapq
import itertools
from math import sqrt

import numpy as np
import holoviews as hv


class BaseLearner(metaclass=abc.ABCMeta):
    """Base class for algorithms for learning a function 'f: X → Y'

    Attributes
    ----------
    function : callable: X → Y
        The function to learn.
    data : dict: X → Y
        'function' evaluated at certain points.
        The values can be 'None', which indicates that the point
        will be evaluated, but that we do not have the result yet.

    Subclasses may define a 'plot' method that takes no parameters
    and returns a holoviews plot.
    """
    def __init__(self, function):
        self.data = {}
        self.function = function

    def add_data(self, xvalues, yvalues):
        """Add data to the learner.

        Parameters
        ----------
        xvalues : value from the function domain, or iterable of such
            Values from the domain of the learned function.
        yvalues : value from the function image, or iterable of such
            Values from the range of the learned function, or None.
            If 'None', then it indicates that the value has not yet
            been computed.
        """
        try:
            for x, y in zip(xvalues, yvalues):
                self.add_point(x, y)
        except TypeError:
            self.add_point(xvalues, yvalues)

    def add_point(self, x, y):
        """Add a single datapoint to the learner."""
        self.data[x] = y

    def remove_unfinished(self):
        """Remove uncomputed data from the learner."""
        self.data = {k: v for k, v in self.data.items() if v is not None}

    @abc.abstractmethod
    def loss(self, real=True):
        """Return the loss for the current state of the learner.

        Parameters
        ----------
        expected : bool, default: False
            If True, return the "expected" loss, i.e. the
            loss including the as-yet unevaluated points
            (possibly by interpolation).
        """

    def choose_points(self, n, add_data=True):
        """Choose the next 'n' points to evaluate.

        Parameters
        ----------
        n : int
            The number of points to choose.
        add_data : bool, default: True
            If True, add the chosen points to this
            learner's 'data' with 'None' for the 'y'
            values. Set this to False if you do not
            want to modify the state of the learner.
        """
        points = self._choose_points(n)
        if add_data:
            self.add_data(points, itertools.repeat(None))
        return points

    @abc.abstractmethod
    def _choose_points(self, n):
        """Choose the next 'n' points to evaluate.

        Should be overridden by subclasses.

        Parameters
        ----------
        n : int
            The number of points to choose.
        """


class Learner1D(BaseLearner):
    """Learns and predicts a function 'f:ℝ → ℝ'.

    Description
    -----------
    Answers questions like:
    * "How much data do you need to get 2% accuracy?"
    * "What is the current status?"
    * "If I give you n data points, which ones would you like?"
    (initialise/request/promise/put/describe current state)

    """

    def __init__(self, function, bounds):
        super().__init__(function)

        # A dict storing the loss function for each interval x_n.
        self.losses = {}
        self.real_losses = {}

        self.real_data = {}
        self.interp_data = {}

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self.neighbors = {}
        self.real_neighbors = {}

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [list(bounds), [np.inf, -np.inf]]

        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [bounds[1] - bounds[0], 0]
        self._oldscale = [bounds[1] - bounds[0], 0]

        self.bounds = list(bounds)

    def interval_loss(self, x_left, x_right, real=False):
        """Calculate loss in the interval x_left, x_right.

        Currently returns the rescaled length of the interval. If one of the
        y-values is missing, returns 0 (so the intervals with missing data are
        never touched. This behavior should be improved later.
        """
        data = self.real_data if real else self.interp_data
        y_right, y_left = data[x_right], data[x_left]
        if self._scale[1] == 0:
            return np.inf
        else:
            return sqrt(((x_right - x_left) / self._scale[0])**2 +
                        ((y_right - y_left) / self._scale[1])**2)

    def loss(self, real=True):
        losses = self.real_losses if real else self.losses

        if len(losses) == 0:
            return float('inf')
        else:
            return max(losses.values())

    def update_neighbors_and_losses(self, x, y, real=False):
        # Update the neighbors.
        neighbors = self.real_neighbors if real else self.neighbors
        if x not in neighbors:  # The point is new
            xvals = sorted(neighbors)
            pos = np.searchsorted(xvals, x)
            neighbors[None] = [None, None]  # To reduce the number of condititons.
            x_lower = xvals[pos-1] if pos != 0 else None
            x_upper = xvals[pos] if pos != len(xvals) else None

            neighbors[x] = [x_lower, x_upper]
            neighbors[x_lower][1] = x
            neighbors[x_upper][0] = x
            del neighbors[None]

        # Update the scale.
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        if real:
            self._bbox[1][0] = min(self._bbox[1][0], y)
            self._bbox[1][1] = max(self._bbox[1][1], y)

        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

        if not real:
            self.interpolate()

        # Update the losses.
        losses = self.real_losses if real else self.losses
        x_lower, x_upper = neighbors[x]
        if x_lower is not None:
            losses[x_lower, x] = self.interval_loss(x_lower, x, real)
        if x_upper is not None:
            losses[x, x_upper] = self.interval_loss(x, x_upper, real)
        try:
            del losses[x_lower, x_upper]
        except KeyError:
            pass

        # If the scale has doubled, recompute all losses.
        # Can only happen when `real`.
        if real:
            if self._scale > self._oldscale * 2:
                    self.real_losses = {key: self.interval_loss(*key, real)
                                        for key in self.real_losses}
                    self.losses = {key: self.interval_loss(*key, real)
                                        for key in self.losses}
                self._oldscale = self._scale

    def add_point(self, x, y):
        super().add_point(x, y)
        self.update_neighbors_and_losses(x, y, real=False)
        real = y is not None
        if real:
            self.real_data[x] = y
            self.update_neighbors_and_losses(x, y, real=True)


    def _choose_points(self, n=10):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        for bound in self.bounds:
            if bound not in self.data:
                if n == 1:
                    return [bound]
                else:
                    return self.bounds

        def points(x, n):
            return list(np.linspace(x[0], x[1], n, endpoint=False)[1:])

        # Calculate how many points belong to each interval.
        quals = [(-loss, x_range, 1) for (x_range, loss) in
                 self.losses.items()]

        heapq.heapify(quals)

        for point_number in range(n):
            quality, x, n = quals[0]
            heapq.heapreplace(quals, (quality * n / (n + 1), x, n + 1))

        xs = sum((points(x, n) for quality, x, n in quals), [])
        return xs

    def remove_unfinished(self):
        super().remove_unfinished()
        self.losses = self.real_losses
        self.neighbors = self.real_neighbors

        # Update the scale.
        self._bbox[0][0] = min(self.data.keys())
        self._bbox[0][1] = max(self.data.keys())
        self._bbox[1][0] = min(self.data.values())
        self._bbox[1][1] = max(self.data.values())
        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

    def interpolate(self):
        xdata = []
        ydata = []
        xdata_unfinished = []
        self.interp_data = {}

        for x in sorted(self.data):
            y = self.data[x]
            if y is None:
                xdata_unfinished.append(x)
            else:
                xdata.append(x)
                ydata.append(y)
                self.interp_data[x] = y

        if len(ydata) == 0:
            ydata_unfinished = (0, ) * len(xdata_unfinished)
        else:
            ydata_unfinished = np.interp(xdata_unfinished, xdata, ydata)

        for x, y in zip(xdata_unfinished, ydata_unfinished):
            self.interp_data[x] = y

    def plot(self):
            xy = [(k, v)
                  for k, v in sorted(self.data.items()) if v is not None]
            if not xy:
                return hv.Scatter([])
            x, y = np.array(xy, dtype=float).transpose()
            return hv.Scatter((x, y))
