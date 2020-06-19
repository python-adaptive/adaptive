# -*- coding: utf-8 -*-

import sys
import itertools
import math
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import sortedcollections
import sortedcontainers
from scipy.stats import t as tstud
import random

from adaptive.learner.learner1D import Learner1D, _get_neighbors_from_list, loss_manager, _get_intervals
from adaptive.notebook_integration import ensure_holoviews

class AverageLearner1D(Learner1D):
    """Learns and predicts a noisy function 'f:ℝ → ℝ^N'.

    New parameters (wrt Learner1D)
    ------------------------------
    delta : float
        This parameter controls the resampling condition. A point is resampled
        if its uncertainty is larger than delta times the smallest neighboring
        interval.
        We strongly recommend 0 < delta <= 1.
    alfa : float (0 < alfa < 1)
        The size of the interval of confidence of the estimate of the mean
        is 1-2*alfa.
    min_samples : int (0 < min_samples)
        Minimum number of samples at each point x. Each new point is initially
        sampled at least min_samples times.
    neighbor_sampling : float (0 < neighbor_sampling <= 1)
        Each new point is initially sampled at least a (neighbor_sampling*100)%
        of the average number of samples of its neighbors.
    max_samples : int (min_samples < max_samples)
        Maximum number of samples at each point x.
    min_Delta_g : float (0 <= min_Delta_g)
        Minimum uncertainty. If the uncertainty at a certain point is below this
        threshold, the point will not be resampled again.

        We recommend to keep alfa=0.005.
    """
    def __init__(self, function, bounds, loss_per_interval = None, delta = 0.2, alfa = 0.005,
                 min_samples = 50, neighbor_sampling = 0.3, max_samples = np.inf, min_Delta_g = 0):
        # Asserts
        assert delta>0, 'delta should be positive (0 < delta <= 1).'
        assert alfa>0 and alfa<1, 'alfa should be positive (0 < alfa < 1).'
        assert min_samples>0, 'min_samples should be positive.'
        assert neighbor_sampling>0, 'neighbor_sampling should be positive (0 < neighbor_sampling <= 1).'
        assert max_samples>min_samples, 'max_samples should be larger than min_samples.'

        super().__init__(function, bounds, loss_per_interval)

        self.delta = delta
        self.alfa = alfa
        self.min_samples = min_samples
        self.min_Delta_g = min_Delta_g
        self.max_samples = max_samples
        self.neighbor_sampling = neighbor_sampling

        self._data_samples = sortedcontainers.SortedDict() # This SortedDict contains all samples f(x) for each
                                                           # point x in the form {x0:[f_0(x0), f_1(x0), ...], ...}
        self._number_samples = sortedcontainers.SortedDict() # This SortedDict contains the number of samples taken
                                                             # at each point x in the form {x0: n0, x1: n1, ...}
        self._undersampled_points = set() # This set contains the points x that have less than min_samples
                                          # samples or less than a (neighbor_sampling*100)% of their neighbors
        self._error_in_mean = decreasing_dict_initializer() # This SortedDict contains the error in the estimate of the
                                                          # mean at each point x in the form {x0: error(x0), ...}
        self._distances = decreasing_dict_initializer() # Distance between two neighboring points in the
                                                           # form {xi: ((xii-xi)^2 + (yii-yi)^2)^0.5, ...}
        self._rescaled_error_in_mean = decreasing_dict_initializer() # {xii: _error_in_mean[xii]/min(_distances[xi],
                                                                   #  _distances[xii], ...}

    @property
    def total_samples(self):
        '''Returns the total number of samples'''
        if not len(self.data):
            return 0
        else:
            _, ns = zip(*self._number_samples.items())
            return sum(ns)

    def ask(self, n, tell_pending=True):
        """Return 'n' points that are expected to maximally reduce the loss."""
        # If some point is undersampled, resample it
        if len(self._undersampled_points):
            for x in self._undersampled_points: # This is to get an element from the set
                break
            points, loss_improvements = self._ask_for_more_samples(x,n)
        # If less than 2 points were sampled, sample a new one
        elif not self.data.__len__() or self.data.__len__() == 1:
            points, loss_improvements = self._ask_for_new_point(n)
        # Else, check the resampling condition
        else:
            if len(self._rescaled_error_in_mean): # This is in case _rescaled_error_in_mean is empty (e.g. when sigma=0)
                x, resc_error = self._rescaled_error_in_mean.peekitem(0)
                # Resampling condition
                if (resc_error > self.delta):
                        points, loss_improvements = self._ask_for_more_samples(x,n)
                else:
                        points, loss_improvements = self._ask_for_new_point(n)
            else:
                points, loss_improvements = self._ask_for_new_point(n)

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _ask_for_more_samples(self,x,n):
        '''When asking for n points, the learner returns n times an existing point
           to be resampled, since in general n << min_samples and this point will
           need to be resampled many more times'''
        points = [x] * n
        loss_improvements = [0] * n # We set the loss_improvements of resamples to 0
        return points, loss_improvements

    def _ask_for_new_point(self,n):
        '''When asking for n new points, the learner returns n times a single
           new point, since in general n << min_samples and this point will need
           to be resampled many more times'''
        points, loss_improvements = self._ask_points_without_adding(1)
        points = points * n
        loss_improvements = loss_improvements + [0] * (n-1)
        return points, loss_improvements

    def tell_pending(self, x):
        if x in self.data:
            self.pending_points.add(x) # Note that a set cannot contain duplicates
            return
        else:
            self.pending_points.add(x) # Note that a set cannot contain duplicates
            self._update_neighbors(x, self.neighbors_combined)
            self._update_losses(x, real=False)

    def tell(self, x, y):
        if y is None:
            raise TypeError(
                "Y-value may not be None, use learner.tell_pending(x)"
                "to indicate that this value is currently being calculated"
            )
        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        if not self.data.__contains__(x):
            self._update_data(x, y, 'new')
            self._update_data_structures(x, y, 'new')
        else:
            self._update_data(x, y, 'resampled')
            self._update_data_structures(x, y, 'resampled')
        self.pending_points.discard(x)

    def _update_rescaled_error_in_mean(self, x, point_type):
        '''Updates self._rescaled_error_in_mean; point_type must be "new" or
           "resampled". '''
        assert point_type=='new' or point_type=='resampled', 'point_type must be "new" or "resampled"'
        # Update neighbors
        xleft, xright = self.neighbors[x]
        if xleft is None and xright is None:
            return
        if (xleft is None):
            dleft = self._distances[x]
        else:
            dleft = self._distances[xleft]
            if self._rescaled_error_in_mean.__contains__(xleft):
                xll = self.neighbors[xleft][0]
                if xll is None:
                    self._rescaled_error_in_mean[xleft] = self._error_in_mean[xleft] / self._distances[xleft]
                else:
                    self._rescaled_error_in_mean[xleft] = self._error_in_mean[xleft] / min(self._distances[xll],
                                                                                           self._distances[xleft])
        if (xright is None):
            dright = self._distances[xleft]
        else:
            dright = self._distances[x]
            if self._rescaled_error_in_mean.__contains__(xright):
                xrr = self.neighbors[xright][1]
                if xrr is None:
                    self._rescaled_error_in_mean[xright] = self._error_in_mean[xright] / self._distances[x]
                else:
                    self._rescaled_error_in_mean[xright] = self._error_in_mean[xright] / min(self._distances[x],
                                                                                             self._distances[xright])
        # Update x
        if point_type=='resampled':
            self._rescaled_error_in_mean[x] = self._error_in_mean[x] / min(dleft,dright)
        return

    def _update_data(self, x, y, point_type):
        assert point_type=='new' or point_type=='resampled', 'point_type must be "new" or "resampled"'
        if point_type=='new':
            self.data[x] = y
        elif point_type=='resampled':
            n = len(self._data_samples[x])
            new_average = self.data[x]*n/(n+1) + y/(n+1)
            self.data[x] = new_average

    def _update_data_structures(self, x, y, point_type):
        assert point_type=='new' or point_type=='resampled', 'point_type must be "new" or "resampled"'

        if point_type=='new':
            self._data_samples[x] = [y]

            if not self.bounds[0] <= x <= self.bounds[1]:
                return

            self._update_neighbors(x, self.neighbors_combined)
            self._update_neighbors(x, self.neighbors)
            self._update_scale(x, y)
            self._update_losses(x, real=True)

            # If the scale has increased enough, recompute all losses.
            if self._scale[1] > self._recompute_losses_factor * self._oldscale[1]:
                for interval in reversed(self.losses):
                    self._update_interpolated_loss_in_interval(*interval)

                self._oldscale = deepcopy(self._scale)

            self._number_samples[x] = 1
            self._undersampled_points.add(x)
            self._error_in_mean[x] = np.inf
            self._rescaled_error_in_mean[x] = np.inf
            self._update_distances(x)
            self._update_rescaled_error_in_mean(x, 'new')

        elif point_type=='resampled':
            self._data_samples[x].append(y)

            self._number_samples[x] = self._number_samples[x]+1
            n = self._number_samples[x]

            if (x in self._undersampled_points) and (n >= self.min_samples):
                xleft, xright = self.neighbors[x]
                n = self._number_samples[x]
                if xleft and xright:
                    nneighbor = 0.5*(self._number_samples[xleft] + self._number_samples[xright])
                elif xleft:
                    nneighbor = self._number_samples[xleft]
                elif xright:
                    nneighbor = self._number_samples[xright]
                else:
                    nneighbor = 0
                if n > self.neighbor_sampling * nneighbor:
                    self._undersampled_points.discard(x)

            # We compute the error in the estimation of the mean as
            # the std of the mean multiplied by a t-Student factor to ensure that
            # the mean value lies within the correct interval of confidence
            y_avg = self.data[x]
            variance_in_mean = sum( [(yj-y_avg)**2 for yj in self._data_samples[x]] )/(n-1)
            t_student = tstud.ppf(1.0 - self.alfa, df=n-1)
            self._error_in_mean[x] = t_student*(variance_in_mean/n)**0.5

            self._update_distances(x)

            self._update_rescaled_error_in_mean(x,'resampled')

            if (self._rescaled_error_in_mean.__contains__(x)
                and (self._error_in_mean[x] <= self.min_Delta_g or self._number_samples[x] >= self.max_samples)):
                _ = self._rescaled_error_in_mean.pop(x)

            # We also need to update scale and losses
            super()._update_scale(x, y)
            self._update_losses_resampling(x, real=True) # REVIEW

            '''Is the following necessary?'''
            # If the scale has increased enough, recompute all losses.
            if self._scale[1] > self._recompute_losses_factor * self._oldscale[1]:
                for interval in reversed(self.losses):
                    self._update_interpolated_loss_in_interval(*interval)
                self._oldscale = deepcopy(self._scale)

    def _update_distances(self, x):
        neighbors = self.neighbors[x]
        if neighbors[0] is not None:
        #    self._distances[neighbors[0]] = x-neighbors[0]
            self._distances[neighbors[0]] = ((x-neighbors[0])**2 + (self.data[x]-self.data[neighbors[0]])**2)**0.5
        if neighbors[1] is not None:
            self._distances[x] = ((neighbors[1]-x)**2 + (self.data[neighbors[1]]-self.data[x])**2)**0.5
        return

    def _update_losses_resampling(self, x, real=True):
        """Update all losses that depend on x, whenever the new point is a
           re-sampled point"""
        # (x_left, x_right) are the "real" neighbors of 'x'.
        x_left, x_right = self._find_neighbors(x, self.neighbors)
        # (a, b) are the neighbors of the combined interpolated
        # and "real" intervals.
        a, b = self._find_neighbors(x, self.neighbors_combined)

        if real:
            for ival in _get_intervals(x, self.neighbors, self.nth_neighbors):
                self._update_interpolated_loss_in_interval(*ival)
        elif x_left is not None and x_right is not None:
            # 'x' happens to be in between two real points,
            # so we can interpolate the losses.
            dx = x_right - x_left
            loss = self.losses[x_left, x_right]
            self.losses_combined[a, x] = (x - a) * loss / dx
            self.losses_combined[x, b] = (b - x) * loss / dx

        # (no real point left of x) or (no real point right of a)
        left_loss_is_unknown = (x_left is None) or (not real and x_right is None)
        if (a is not None) and left_loss_is_unknown:
            self.losses_combined[a, x] = float("inf")

        # (no real point right of x) or (no real point left of b)
        right_loss_is_unknown = (x_right is None) or (not real and x_left is None)
        if (b is not None) and right_loss_is_unknown:
            self.losses_combined[x, b] = float("inf")

    def tell_many(self, xs, ys, *, force=False):
        '''The data should be given as:
                - {x_i: y_i} (only the mean at each point), in which case the
		  number of samples is assumed to be 1 and the error 0 for all
		  data points. These points will not be included in
                  _rescaled_error_in_mean and therefore will not be resampled.
                - {x_i: [y_i0, y_i1, ...]} (all data samples at each point).'''

        for y in ys:
            # If data_samples is given:
            if isinstance(y, list):
                self._data_samples.update(zip(xs, ys))
                print(self._undersampled_points)

                super().tell_many(xs, [np.mean(y) for y in ys]) # self.data is updated here
                yslen = []
                ysavg = []
                print(self._undersampled_points)
                for ii in np.arange(len(ys)):
                    x = xs[ii]
                    y = ys[ii]
                    y_avg = np.mean(y)
                    n = len(y)
                    yslen.append(n)
                    ysavg.append(y_avg)
                    # We include the point in _undersampled_points if there are
                    # less than min_samples samples, disregarding neighbor_sampling.
                    # super().tell_many() sometimes calls self.tell(), which includes
                    # x in _undersampled_points, so we may need to remove it.
                    if n < self.min_samples:
                        self._undersampled_points.add(x)
                    elif n > self.min_samples:
                        self._undersampled_points.discard(x)
                    # _error_in_mean:
                    variance_in_mean = sum( [(yj-y_avg)**2 for yj in y] )/(n-1)
                    t_student = tstud.ppf(1.0 - self.alfa, df=n-1)
                    self._error_in_mean[x] = t_student*(variance_in_mean/n)**0.5
                    # _update_distances:
                    self._update_distances(x)

                self._number_samples.update(zip(xs, yslen))
                print(self._undersampled_points)

                for x in xs:
                    if self._number_samples[x] == 1:
                        self._rescaled_error_in_mean[x] = np.inf
                    elif self._number_samples[x] < self.max_samples:
                        self._update_rescaled_error_in_mean(x, 'resampled')
                self._data_samples.update(zip(xs, ys))

            # If data is given:
            else:
                super().tell_many(xs, ys) # self.data is updated here
                self._data_samples.update(zip(xs, ys))
                self._number_samples.update(zip(xs, [1]*len(xs)))
                self._error_in_mean.update(zip(xs, [0]*len(xs)))
                for x in xs:
                    self._update_distances(x)

            break

        return

    def plot(self):
        """Returns a plot of the evaluated data with error bars.

        Returns
        -------
        plot : `holoviews.element.Scatter` (if vdim=1)\
               else `holoviews.element.Path`
            Plot of the evaluated data.
        """
        hv = ensure_holoviews()
        if not self.data:
            p = hv.Scatter([]) * hv.ErrorBars([]) * hv.Path([])
        elif not self.vdim > 1:
            p = hv.Scatter(self.data) * hv.ErrorBars([(x, self.data[x], self._error_in_mean[x]) for x in self.data]) * hv.Path([])
        else:
            raise Exception('plot() not implemented for vector functions.')
            xs, ys = zip(*sorted(self.data.items()))
            p = hv.Path((xs, ys)) * hv.Scatter([])

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (self.bounds[1] - self.bounds[0])
        plot_bounds = (self.bounds[0] - margin, self.bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))

def decreasing_dict_initializer():
    '''This initialization orders the dictionary from large to small values'''
    def sorting_rule(key, value):
        return -value
    return sortedcollections.ItemSortedDict(sorting_rule, sortedcontainers.SortedDict())
