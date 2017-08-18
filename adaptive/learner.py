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
    def loss(self):
        pass

    def choose_points(self, n, add_data=True):
        points = self._choose_points(n)
        if add_data:
            self.add_data(points, itertools.repeat(None))
        return points

    @abc.abstractmethod
    def _choose_points(self, n):
        pass

    @abc.abstractmethod
    def interpolate(self):
        pass


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

    def __init__(self, function):
        super().__init__(function)

        # A dict storing the loss function for each interval x_n.
        self.losses = {}

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self.neighbors = {}

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [[np.inf, -np.inf], [np.inf, -np.inf]]
        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [0, 0]
        self._oldscale = [0, 0]

    def interval_loss(self, x_left, x_right):
        """Calculate loss in the interval x_left, x_right.

        Currently returns the rescaled length of the interval. If one of the
        y-values is missing, returns 0 (so the intervals with missing data are
        never touched. This behavior should be improved later.
        """
        y_right, y_left = self.interp_data[x_right], self.interp_data[x_left]
        return sqrt(((x_right - x_left) / self._scale[0])**2 +
                    ((y_right - y_left) / self._scale[1])**2)

    def loss(self):
        if len(self.losses) == 0:
            return float('inf')
        else:
            return max(self.losses.values())

    def add_point(self, x, y):
        super().add_point(x, y)

        # Update the scale.
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        if y is not None:
            self._bbox[1][0] = min(self._bbox[1][0], y)
            self._bbox[1][1] = max(self._bbox[1][1], y)
        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

    def _choose_points(self, n=10):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        self.interpolate()  # Apply new interpolation step if new results

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
        # Update the scale.
        self._bbox[0][0] = min(self.data.keys())
        self._bbox[0][1] = max(self.data.keys())
        self._bbox[1][0] = min(self.data.values())
        self._bbox[1][1] = max(self.data.values())
        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

        self.interpolate()

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

        self.neighbors = {}
        xdata_sorted = sorted(self.interp_data)
        for i, x in enumerate(xdata_sorted):
            if i == 0:
                self.neighbors[x] = [None, xdata_sorted[1]]
            elif i == len(xdata_sorted) - 1:
                self.neighbors[x] = [xdata_sorted[i-1], None]
            else:
                self.neighbors[x] = [xdata_sorted[i-1], xdata_sorted[i+1]]

        self.losses = {}
        for x, (x_left, x_right) in self.neighbors.items():
            if x_left is not None:
                self.losses[(x_left, x)] = self.interval_loss(x_left, x)
            if x_right is not None:
                self.losses[x, x_right] = self.interval_loss(x, x_right)
            try:
                del self.losses[x_left, x_right]
            except KeyError:
                pass

    def plot(self):
            xy = [(k, v)
                  for k, v in sorted(self.data.items()) if v is not None]
            if not xy:
                return hv.Scatter([])
            x, y = np.array(xy, dtype=float).transpose()
            return hv.Scatter((x, y))
