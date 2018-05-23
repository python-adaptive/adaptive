# -*- coding: utf-8 -*-
from copy import deepcopy
import heapq
import itertools
import math

import numpy as np
import sortedcontainers

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner

from skopt import Optimizer


class SKOptLearner(Optimizer, BaseLearner):
    """Learn a function minimum using 'skopt.Optimizer'.

    This is an 'Optimizer' from 'scikit-optimize',
    with the necessary methods added to make it conform
    to the 'adaptive' learner interface.

    Parameters
    ----------
    function : callable
        The function to learn.
    **kwargs :
        Arguments to pass to 'skopt.Optimizer'.
    """

    def __init__(self, function, **kwargs):
        self.function = function
        super().__init__(**kwargs)

    def add_point(self, x, y):
        if y is not None:
            # 'skopt.Optimizer' takes care of points we
            # have not got results for.
            self.tell([x], y)

    def remove_unfinished(self):
        pass

    def loss(self, real=True):
        if not self.models:
            return np.inf
        else:
            model = self.models[-1]
            # Return the in-sample error (i.e. test the model
            # with the training data). This is not the best
            # estimator of loss, but it is the cheapest.
            return 1 / model.score(self.Xi, self.yi)

    def choose_points(self, n, add_data=True):
        points = self.ask(n)
        if self.space.n_dims > 1:
            return points, [np.inf] * n
        else:
            return [p[0] for p in points], [np.inf] * n

    @property
    def npoints(self):
        return len(self.Xi)

    def plot(self):
        hv = ensure_holoviews()
        if self.space.n_dims > 1:
            raise ValueError('Can only plot 1D functions')
        bounds = self.space.bounds[0]
        if not self.Xi:
            p = hv.Scatter([]) * hv.Area([])
        else:
            scatter = hv.Scatter(([p[0] for p in self.Xi], self.yi))
            if self.models:
                # Plot 95% confidence interval as colored area around points
                model = self.models[-1]
                xs = np.linspace(*bounds, 201)
                y_pred, sigma = model.predict(np.atleast_2d(xs).transpose(),
                                              return_std=True)
                area = hv.Area(
                    (xs, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma),
                    vdims=['y', 'y2'],
                ).opts(style=dict(alpha=0.5, line_alpha=0))
            else:
                area = hv.Area([])
            p = scatter * area

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (bounds[1] - bounds[0])
        plot_bounds = (bounds[0] - margin, bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))
