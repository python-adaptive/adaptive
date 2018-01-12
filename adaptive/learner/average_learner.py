# -*- coding: utf-8 -*-
import itertools
from math import sqrt

import numpy as np

from .base_learner import BaseLearner

class AverageLearner(BaseLearner):
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

    def __init__(self, function, atol=None, rtol=None):
        if atol is None and rtol is None:
            raise Exception('At least one of `atol` and `rtol` should be set.')
        if atol is None:
            atol = np.inf
        if rtol is None:
            rtol = np.inf

        self.data = {}
        self.function = function
        self.atol = atol
        self.rtol = rtol
        self.n = 0
        self.sum_f = 0
        self.sum_f_sq = 0

    @property
    def n_requested(self):
        return len(self.data)

    def choose_points(self, n, add_data=True):
        points = list(range(self.n_requested, self.n_requested + n))
        loss_improvements = [self.loss()] * n
        if add_data:
            self.add_data(points, itertools.repeat(None))
        return points, loss_improvements

    def add_point(self, n, value):
        value_is_new = not (n in self.data and value == self.data[n])
        self.data[n] = value
        if value is not None:
            if value_is_new:
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
        import holoviews as hv
        vals = [v for v in self.data.values() if v is not None]
        if not vals:
            return hv.Histogram([[], []])
        num_bins = int(max(5, sqrt(self.n)))
        vals = hv.Points(vals)
        return hv.operation.histogram(vals, num_bins=num_bins, dimension=1)
