# -*- coding: utf-8 -*-
import abc
from copy import deepcopy as copy
import functools
import heapq
import itertools
from math import sqrt, isinf

import sortedcontainers
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

    @abc.abstractmethod
    def add_point(self, x, y):
        """Add a single datapoint to the learner."""
        pass

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
        self.data[n] = value
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

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single real parameter and
        return a real number.
    bounds : pair of reals
        The bounds of the interval on which to learn 'function'.
    """

    def __init__(self, function, bounds):
        super().__init__(function)

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
        self._oldscale = copy(self._scale)

        self.bounds = list(bounds)

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
        if self._scale[1] == 0:
            return np.inf
        else:
            return sqrt(((x_right - x_left) / self._scale[0])**2 +
                        ((y_right - y_left) / self._scale[1])**2)

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

    def loss_improvement(self, points):
        current_loss = self.loss(real=False)

        data_interp = self.interpolate(points)
        data = {**self.data_combined, **data_interp}

        # Create a new losses and neighbors dict
        neighbors = copy(self.neighbors_combined)
        losses = copy(self.losses_combined)
        for x in points:
            self.update_neighbors(x, neighbors)
            self.update_losses(x, data, neighbors, losses)

        # Calculate the loss improvement
        if len(losses) == 0:
            return float('inf')
        else:
            loss = max(losses.values())
            if isinf(loss):
                return float('inf')
            else:
                return current_loss - loss

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
            self._bbox[1][0] = min(self._bbox[1][0], y)
            self._bbox[1][1] = max(self._bbox[1][1], y)

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

        if real:
            # If the scale has doubled, recompute all losses.
            if self._scale > self._oldscale * 2:
                    self.losses = {key: self.interval_loss(*key, self.data)
                                   for key in self.losses}
                    self.losses_combined = {key: self.interval_loss(*key,
                        self.data_combined) for key in self.losses_combined}
                    self._oldscale = self._scale


    def _choose_points(self, n=10):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        if n == 0:
            return []

        # If the bounds have not been chosen yet, we choose them first.
        xs = []
        for bound in self.bounds:
            if bound not in self.data and bound not in self.data_interp:
                xs.append(bound)
        # Ensure we return exactly 'n' points.
        if xs:
            if n <= 2:
                return xs[:n]
            else:
                return np.linspace(*self.bounds, n)

        def points(x, n):
            return list(np.linspace(x[0], x[1], n, endpoint=False)[1:])

        # Calculate how many points belong to each interval.
        quals = [(-loss, x_range, 1) for (x_range, loss) in
                 self.losses_combined.items()]

        heapq.heapify(quals)

        for point_number in range(n):
            quality, x, n = quals[0]
            heapq.heapreplace(quals, (quality * n / (n + 1), x, n + 1))

        xs = sum((points(x, n) for quality, x, n in quals), [])
        return xs

    def interpolate(self, extra_points=None):
        xs = list(self.data.keys())
        ys = list(self.data.values())
        xs_unfinished = list(self.data_interp.keys())

        if extra_points is not None:
            xs_unfinished += extra_points

        if len(ys) == 0:
            interp_ys = (0,) * len(xs_unfinished)
        else:
            interp_ys = np.interp(xs_unfinished, xs, ys)

        data_interp = {x: y for x, y in zip(xs_unfinished, interp_ys)}

        return data_interp

    def plot(self):
            if self.data:
                return hv.Scatter(self.data)
            else:
                return hv.Scatter([])

    def remove_unfinished(self):
        self.data_interp = {}
        self.losses = copy(self.losses_combined)
        self.neighbors = copy(self.neighbors_combined)


def dispatch(child_functions, arg):
    index, x = arg
    return child_functions[index](x)


class BalancingLearner(BaseLearner):
    """Choose the optimal points from a set of learners.

    Parameters
    ----------
    learners : sequence of BaseLearner
        The learners from which to choose. These must all have the same type.

    Notes
    -----
    This learner compares the 'loss' calculated from the "child" learners.
    This requires that the 'loss' from different learners *can be meaningfully
    compared*. For the moment we enforce this restriction by requiring that
    all learners are the same type but (depending on the internals of the
    learner) it may be that the loss cannot be compared *even between learners
    of the same type*. In this case the BalancingLearner will behave in an
    undefined way.
    """
    def __init__(self, learners):
        self.learners = learners

        # Naively we would make 'function' a method, but this causes problems
        # when using executors from 'concurrent.futures' because we have to
        # pickle the whole learner.
        self.function = functools.partial(dispatch, [l.function for l
                                                     in self.learners])

        if len(set(learner.__class__ for learner in self.learners)) > 1:
            raise Exception('A BalacingLearner can handle only one type'
                            'of learners.')

    def choose_points(self, n, add_data=True):
        points = self._choose_points(n)
        if add_data:
            for index, point in points:
                self.learners[index].add_data(point, None)
        return points

    def _choose_points(self, n):
        """Choses points of learners.

        Note: Can only return up to len(self.learners) number of points."""
        n = min(len(self.learners), n)

        loss_improvements = []
        pairs = []
        for index, learner in enumerate(self.learners):
            points = learner.choose_points(n=1, add_data=False)
            loss_improvements.append(learner.loss_improvement(points))
            pairs.append((index, points[0]))

        # If no real data availible yet, choose arcording to
        # the length of outstanding jobs.
        if not any(len(L.data) for L in self.learners):
            loss_improvements = [-len(L.data_interp) for L in self.learners]

        pairs = [p for _, p in sorted(zip(loss_improvements, pairs))][::-1]

        return pairs[:n]

    def add_point(self, x, y):
        index, x = x
        self.learners[index].add_point(x, y)

    def loss(self, real=True):
        return max(learner.loss(real) for learner in self.learners)

    def plot(self, index):
        return self.learners[index].plot()

    def remove_unfinished(self):
        """Remove uncomputed data from the learners."""
        for learner in self.learners:
            learner.remove_unfinished()
