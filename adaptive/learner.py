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

    @abc.abstractmethod
    def remove_unfinished(self):
        """Remove uncomputed data from the learner."""
        pass

    @abc.abstractmethod
    def loss(self, real=True):
        """Return the loss for the current state of the learner.

        Parameters
        ----------
        real : bool, default: True
            If False, return the "expected" loss, i.e. the
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

class AverageLearner(BaseLearner):
    def __init__(self, function, atol=None, rtol=None):
        """A naive implementation of adaptive computing of averages.

        The learned function must depend on an integer input variable that
        represents the source of randomness.

        Parameters:
        -----------
        atol : float
            Desired absolute tolerance
        rtol : float
            Desired relative tolerance
        """
        super().__init__(function)

        if atol is None and rtol is None:
            raise Exception('At least one of `atol` and `rtol` should be set.')
        if atol is None:
            atol = np.inf
        if rtol is None:
            rtol = np.inf

        self.function = function
        self.atol = atol
        self.rtol = rtol
        self.n = 0
        self.n_requested = 0
        self.sum_f = 0
        self.sum_f_sq = 0

    def _choose_points(self, n=10):
        return list(range(self.n_requested, self.n_requested + n))

    def add_point(self, n, value):
        super().add_point(n, value)
        if value is None:
            self.n_requested += 1
            return
        else:
            self.n += 1
            self.sum_f += value
            self.sum_f_sq += value**2

    @property
    def mean(self):
        return self.sum_f / self.n

    @property
    def std(self):
        n = self.n
        if n < 2:
            return np.inf
        return sqrt((self.sum_f_sq - n * self.mean**2) / (n - 1))

    def loss(self, real=True):
        n = self.n
        if n < 2:
            return np.inf
        standard_error = self.std / sqrt(n if real else self.n_requested)
        return max(standard_error / self.atol,
                   standard_error / abs(self.mean) / self.rtol)

    def remove_unfinished(self):
        """Remove uncomputed data from the learner."""
        pass

    def plot(self):
        vals = [v for v in self.data.values() if v is not None]
        if not vals:
            return hv.Histogram([[], []])
        num_bins = int(max(5, sqrt(self.n)))
        vals = hv.Points(vals)
        return hv.operation.histogram(vals, num_bins=num_bins, dimension=1)


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

    def _interval_loss(self, x_left, x_right, y_right, y_left):
        if self._scale[1] == 0:
            return np.inf
        else:
            return sqrt(((x_right - x_left) / self._scale[0])**2 +
                        ((y_right - y_left) / self._scale[1])**2)

    def interval_loss(self, x_left, x_right, real=False):
        """Calculate loss in the interval x_left, x_right.

        Currently returns the rescaled length of the interval. If one of the
        y-values is missing, returns 0 (so the intervals with missing data are
        never touched. This behavior should be improved later.
        """
        data = self.real_data if real else self.interp_data
        y_right, y_left = data[x_right], data[x_left]
        return self._interval_loss(x_left, x_right, y_right, y_left)

    def loss(self, real=True):
        losses = self.real_losses if real else self.losses

        if len(losses) == 0:
            return float('inf')
        else:
            return max(losses.values())

    def loss_improvement(self, points):
        pass

    def find_neighbors(self, x, real):
        neighbors = self.real_neighbors if real else self.neighbors
        xvals = sorted(neighbors)
        pos = np.searchsorted(xvals, x)
        x_lower = xvals[pos-1] if pos != 0 else None
        x_upper = xvals[pos] if pos != len(xvals) else None
        return x_lower, x_upper

    def update_neighbors_and_losses(self, x, y, real):
        # Update the neighbors.
        neighbors = self.real_neighbors if real else self.neighbors
        if x not in neighbors:  # The point is new
            x_lower, x_upper = self.find_neighbors(x, real)
            neighbors[x] = [x_lower, x_upper]
            neighbors[None] = [None, None]  # To reduce the number of condititons.
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
            self.interp_data = self.interpolate()

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
                    self.real_losses = {key: self.interval_loss(*key, real=True)
                                        for key in self.real_losses}
                    self.losses = {key: self.interval_loss(*key, real=False)
                                   for key in self.losses}
                    self._oldscale = self._scale

    def add_point(self, x, y):
        super().add_point(x, y)
        self.update_neighbors_and_losses(x, y, real=False)

        if y is not None:
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
        self.data = {k: v for k, v in self.data.items() if v is not None}
        # self.losses = self.real_losses
        # self.neighbors = self.real_neighbors

        # Update the scale.
        if self.data:
            self._bbox[0][0] = min(self.data.keys())
            self._bbox[0][1] = max(self.data.keys())
            self._bbox[1][0] = min(self.data.values())
            self._bbox[1][1] = max(self.data.values())
            self._scale = [self._bbox[0][1] - self._bbox[0][0],
                           self._bbox[1][1] - self._bbox[1][0]]

    def interpolate(self, extra_points=None):
        xs = []
        ys = []
        xs_unfinished = [] if extra_points is None else extra_points
        interp_data = {}

        for x in sorted(self.data):
            y = self.data[x]
            if y is None:
                xs_unfinished.append(x)
            else:
                xs.append(x)
                ys.append(y)
                interp_data[x] = y

        if len(ys) == 0:
            interp_ys = (0,) * len(xs_unfinished)
        else:
            interp_ys = np.interp(xs_unfinished, xs, ys)

        for x, y in zip(xs_unfinished, interp_ys):
            interp_data[x] = y

        return interp_data

    def plot(self):
            xy = [(k, v)
                  for k, v in sorted(self.data.items()) if v is not None]
            if not xy:
                return hv.Scatter([])
            x, y = np.array(xy, dtype=float).transpose()
            return hv.Scatter((x, y))
