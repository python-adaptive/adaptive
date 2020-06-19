from copy import deepcopy
from math import hypot

import numpy as np
import scipy.stats
from sortedcollections import ItemSortedDict
from sortedcontainers import SortedDict

from adaptive.learner.learner1D import Learner1D, _get_intervals
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
    alpha : float (0 < alpha < 1)
        The size of the interval of confidence of the estimate of the mean
        is 1-2*alpha. We recommend to keep alpha=0.005.
    neighbor_sampling : float (0 < neighbor_sampling <= 1)
        Each new point is initially sampled at least a (neighbor_sampling*100)%
        of the average number of samples of its neighbors.
    min_samples : int (min_samples > 0)
        Minimum number of samples at each point x. Each new point is initially
        sampled at least min_samples times.
    max_samples : int (min_samples < max_samples)
        Maximum number of samples at each point x.
    min_Delta_g : float (min_Delta_g >= 0)
        Minimum uncertainty. If the uncertainty at a certain point is below this
        threshold, the point will not be resampled again.
    """

    def __init__(
        self,
        function,
        bounds,
        loss_per_interval=None,
        delta=0.2,
        alpha=0.005,
        neighbor_sampling=0.3,
        min_samples=50,
        max_samples=np.inf,
        min_Delta_g=0,
    ):
        # Checks
        for k, v in zip(
            ["delta", "alpha", "neighbor_sampling"], [delta, alpha, neighbor_sampling]
        ):
            if not (0 < v <= 1):
                raise ValueError(f"{k} should be positive (0 < {k} <= 1).")
        if min_samples < 0:
            raise ValueError("min_samples should be positive.")
        if min_samples > max_samples:
            raise ValueError("max_samples should be larger than min_samples.")

        super().__init__(function, bounds, loss_per_interval)

        self.delta = delta
        self.alpha = alpha
        self.min_samples = min_samples
        self.min_Delta_g = min_Delta_g
        self.max_samples = max_samples
        self.neighbor_sampling = neighbor_sampling

        # Contains all samples f(x) for each
        # point x in the form {x0:[f_0(x0), f_1(x0), ...], ...}
        self._data_samples = SortedDict()
        # Contains the number of samples taken
        # at each point x in the form {x0: n0, x1: n1, ...}
        self._number_samples = SortedDict()
        # This set contains the points x that have less than min_samples
        # samples or less than a (neighbor_sampling*100)% of their neighbors
        self._undersampled_points = set()
        # Contains the error in the estimate of the
        # mean at each point x in the form {x0: error(x0), ...}
        self._error_in_mean = decreasing_dict()
        #  Distance between two neighboring points in the
        # form {xi: ((xii-xi)^2 + (yii-yi)^2)^0.5, ...}
        self._distances = decreasing_dict()
        # {xii: _error_in_mean[xii]/min(_distances[xi], _distances[xii], ...}
        self._rescaled_error_in_mean = decreasing_dict()

    @property
    def total_samples(self):
        """Returns the total number of samples"""
        return sum(self._number_samples.values())

    def ask(self, n, tell_pending=True):
        """Return 'n' points that are expected to maximally reduce the loss."""
        # If some point is undersampled, resample it
        if len(self._undersampled_points):
            x = next(iter(self._undersampled_points))
            points, loss_improvements = self._ask_for_more_samples(x, n)
        # If less than 2 points were sampled, sample a new one
        elif len(self.data) <= 1:
            points, loss_improvements = self._ask_for_new_point(n)
        #  Else, check the resampling condition
        else:
            if len(self._rescaled_error_in_mean):
                # This is in case _rescaled_error_in_mean is empty (e.g. when sigma=0)
                x, resc_error = self._rescaled_error_in_mean.peekitem(0)
                # Resampling condition
                if resc_error > self.delta:
                    points, loss_improvements = self._ask_for_more_samples(x, n)
                else:
                    points, loss_improvements = self._ask_for_new_point(n)
            else:
                points, loss_improvements = self._ask_for_new_point(n)

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _ask_for_more_samples(self, x, n):
        """When asking for n points, the learner returns n times an existing point
           to be resampled, since in general n << min_samples and this point will
           need to be resampled many more times"""
        points = [x] * n
        loss_improvements = [0] * n  # We set the loss_improvements of resamples to 0
        return points, loss_improvements

    def _ask_for_new_point(self, n):
        """When asking for n new points, the learner returns n times a single
           new point, since in general n << min_samples and this point will need
           to be resampled many more times"""
        points, loss_improvements = self._ask_points_without_adding(1)
        points = points * n
        loss_improvements = loss_improvements + [0] * (n - 1)
        return points, loss_improvements

    def tell_pending(self, x):
        if x in self.data:
            self.pending_points.add(x)
        else:
            self.pending_points.add(x)
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

        if x not in self.data:
            self._update_data(x, y, "new")
            self._update_data_structures(x, y, "new")
        else:
            self._update_data(x, y, "resampled")
            self._update_data_structures(x, y, "resampled")
        self.pending_points.discard(x)

    def _update_rescaled_error_in_mean(self, x, point_type):
        """Updates self._rescaled_error_in_mean; point_type must be "new" or
           "resampled". """
        #  Update neighbors
        x_left, x_right = self.neighbors[x]
        dists = self._distances
        if x_left is None and x_right is None:
            return

        if x_left is None:
            d_left = dists[x]
        else:
            d_left = dists[x_left]
            if x_left in self._rescaled_error_in_mean:
                xll = self.neighbors[x_left][0]
                norm = dists[x_left] if xll is None else min(dists[xll], dists[x_left])
                self._rescaled_error_in_mean[x_left] = (
                    self._error_in_mean[x_left] / norm
                )

        if x_right is None:
            d_right = dists[x_left]
        else:
            d_right = dists[x]
            if x_right in self._rescaled_error_in_mean:
                xrr = self.neighbors[x_right][1]
                norm = dists[x] if xrr is None else min(dists[x], dists[x_right])
                self._rescaled_error_in_mean[x_right] = (
                    self._error_in_mean[x_right] / norm
                )

        # Update x
        if point_type == "resampled":
            norm = min(d_left, d_right)
            self._rescaled_error_in_mean[x] = self._error_in_mean[x] / norm

    def _update_data(self, x, y, point_type):
        if point_type == "new":
            self.data[x] = y
        elif point_type == "resampled":
            n = len(self._data_samples[x])
            new_average = self.data[x] * n / (n + 1) + y / (n + 1)
            self.data[x] = new_average

    def _update_data_structures(self, x, y, point_type):
        if point_type == "new":
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
            self._update_rescaled_error_in_mean(x, "new")

        elif point_type == "resampled":
            self._data_samples[x].append(y)
            ns = self._number_samples
            ns[x] += 1
            n = ns[x]
            if (x in self._undersampled_points) and (n >= self.min_samples):
                x_left, x_right = self.neighbors[x]
                if x_left is not None and x_right is not None:
                    nneighbor = (ns[x_left] + ns[x_right]) / 2
                elif x_left is not None:
                    nneighbor = ns[x_left]
                elif x_right is not None:
                    nneighbor = ns[x_right]
                else:
                    nneighbor = 0
                if n > self.neighbor_sampling * nneighbor:
                    self._undersampled_points.discard(x)

            # We compute the error in the estimation of the mean as
            # the std of the mean multiplied by a t-Student factor to ensure that
            # the mean value lies within the correct interval of confidence
            y_avg = self.data[x]
            ys = self._data_samples[x]
            self._error_in_mean[x] = self._calc_error_in_mean(ys, y_avg, n)
            self._update_distances(x)
            self._update_rescaled_error_in_mean(x, "resampled")

            if self._error_in_mean[x] <= self.min_Delta_g or n >= self.max_samples:
                self._rescaled_error_in_mean.pop(x, None)

            # We also need to update scale and losses
            super()._update_scale(x, y)
            self._update_losses_resampling(x, real=True)  #  XXX: REVIEW

            # If the scale has increased enough, recompute all losses.
            # We only update the scale considering resampled points, since new
            # points are more likely to be outliers.
            if self._scale[1] > self._recompute_losses_factor * self._oldscale[1]:
                for interval in reversed(self.losses):
                    self._update_interpolated_loss_in_interval(*interval)
                self._oldscale = deepcopy(self._scale)

    def _update_distances(self, x):
        x_left, x_right = self.neighbors[x]
        y = self.data[x]
        if x_left is not None:
            self._distances[x_left] = hypot((x - x_left), (y - self.data[x_left]))
        if x_right is not None:
            self._distances[x] = hypot((x_right - x), (self.data[x_right] - y))

    def _update_losses_resampling(self, x, real=True):
        """Update all losses that depend on x, whenever the new point is a re-sampled point."""
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

    def _calc_error_in_mean(self, ys, y_avg, n):
        variance_in_mean = sum((y - y_avg) ** 2 for y in ys) / (n - 1)
        t_student = scipy.stats.t.ppf(1 - self.alpha, df=n - 1)
        return t_student * (variance_in_mean / n) ** 0.5

    def tell_many(self, xs, ys, *, force=False):
        """The data should be given as:
                - {x_i: y_i} (only the mean at each point), in which case the
                  number of samples is assumed to be 1 and the error 0 for all
                  data points. These points will not be included in
                  _rescaled_error_in_mean and therefore will not be resampled.
                - {x_i: [y_i0, y_i1, ...]} (all data samples at each point)."""
        # If data_samples is given:
        if isinstance(ys[0], list):
            self._data_samples.update(zip(xs, ys))
            ys_avg = np.mean(ys, axis=1)
            super().tell_many(xs, ys_avg)  # self.data is updated here
            ys_len = []
            for x, ys_, y_avg in zip(xs, ys, ys_avg):
                n = len(ys_)
                ys_len.append(n)
                # We include the point in _undersampled_points if there are
                # less than min_samples samples, disregarding neighbor_sampling.
                # super().tell_many() sometimes calls self.tell(), which includes
                # x in _undersampled_points, so we may need to remove it.
                if n < self.min_samples:
                    self._undersampled_points.add(x)
                elif n > self.min_samples:
                    self._undersampled_points.discard(x)
                self._error_in_mean[x] = self._calc_error_in_mean(ys_, y_avg, n)
                self._update_distances(x)

            self._number_samples.update(zip(xs, ys_len))

            for x in xs:
                if self._number_samples[x] == 1:
                    self._rescaled_error_in_mean[x] = np.inf
                elif self._number_samples[x] < self.max_samples:
                    self._update_rescaled_error_in_mean(x, "resampled")
            self._data_samples.update(zip(xs, ys))

        # If data is given:
        else:
            super().tell_many(xs, ys)  # self.data is updated here
            self._data_samples.update(zip(xs, ys))
            self._number_samples.update(zip(xs, [1] * len(xs)))
            self._error_in_mean.update(zip(xs, [0] * len(xs)))
            for x in xs:
                self._update_distances(x)

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
            xs, ys = zip(*sorted(self.data.items()))
            p = (
                hv.Scatter(self.data)
                * hv.ErrorBars(
                    [(x, self.data[x], self._error_in_mean[x]) for x in self.data]
                )
                * hv.Path((xs, ys))
            )
        else:
            raise Exception("plot() not implemented for vector functions.")

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (self.bounds[1] - self.bounds[0])
        plot_bounds = (self.bounds[0] - margin, self.bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))


def decreasing_dict():
    """This initialization orders the dictionary from large to small values"""

    def sorting_rule(key, value):
        return -value

    return ItemSortedDict(sorting_rule, SortedDict())
