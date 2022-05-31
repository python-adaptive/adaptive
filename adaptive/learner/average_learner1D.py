from __future__ import annotations

import math
import sys
from collections import defaultdict
from copy import deepcopy
from math import hypot
from typing import Callable, DefaultDict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.stats
from sortedcollections import ItemSortedDict
from sortedcontainers import SortedDict

from adaptive.learner.learner1D import Learner1D, _get_intervals
from adaptive.notebook_integration import ensure_holoviews
from adaptive.types import Real

Point = Tuple[int, Real]
Points = List[Point]

__all__: list[str] = ["AverageLearner1D"]


class AverageLearner1D(Learner1D):
    """Learns and predicts a noisy function 'f:ℝ → ℝ'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a tuple of ``(seed, x)`` and
        return a real number.
    bounds : pair of reals
        The bounds of the interval on which to learn 'function'.
    loss_per_interval: callable, optional
        A function that returns the loss for a single interval of the domain.
        If not provided, then a default is used, which uses the scaled distance
        in the x-y plane as the loss. See the notes for more details
        of `adaptive.Learner1D` for more details.
    delta : float, optional, default 0.2
        This parameter controls the resampling condition. A point is resampled
        if its uncertainty is larger than delta times the smallest neighboring
        interval.
        We strongly recommend ``0 < delta <= 1``.
    alpha : float (0 < alpha < 1), default 0.005
        The true value of the function at x is within the confidence interval
        ``[self.data[x] - self.error[x], self.data[x] + self.error[x]]`` with
        probability ``1-2*alpha``.
        We recommend to keep ``alpha=0.005``.
    neighbor_sampling : float (0 < neighbor_sampling <= 1), default 0.3
        Each new point is initially sampled at least a (neighbor_sampling*100)%
        of the average number of samples of its neighbors.
    min_samples : int (min_samples > 0), default 50
        Minimum number of samples at each point x. Each new point is initially
        sampled at least min_samples times.
    max_samples : int (min_samples < max_samples), default np.inf
        Maximum number of samples at each point x.
    min_error : float (min_error >= 0), default 0
        Minimum size of the confidence intervals. The true value of the
        function at x is within the confidence interval [self.data[x] -
        self.error[x], self.data[x] + self.error[x]] with
        probability 1-2*alpha.
        If self.error[x] < min_error, then x will not be resampled
        anymore, i.e., the smallest confidence interval at x is
        [self.data[x] - min_error, self.data[x] + min_error].
    """

    def __init__(
        self,
        function: Callable[[tuple[int, Real]], Real],
        bounds: tuple[Real, Real],
        loss_per_interval: None
        | (Callable[[Sequence[Real], Sequence[Real]], float]) = None,
        delta: float = 0.2,
        alpha: float = 0.005,
        neighbor_sampling: float = 0.3,
        min_samples: int = 50,
        max_samples: int = sys.maxsize,
        min_error: float = 0,
    ):
        if not (0 < delta <= 1):
            raise ValueError("Learner requires 0 < delta <= 1.")
        if not (0 < alpha <= 1):
            raise ValueError("Learner requires 0 < alpha <= 1.")
        if not (0 < neighbor_sampling <= 1):
            raise ValueError("Learner requires 0 < neighbor_sampling <= 1.")
        if min_samples < 0:
            raise ValueError("min_samples should be positive.")
        if min_samples > max_samples:
            raise ValueError("max_samples should be larger than min_samples.")

        super().__init__(function, bounds, loss_per_interval)

        self.delta = delta
        self.alpha = alpha
        self.min_samples = min_samples
        self.min_error = min_error
        self.max_samples = max_samples
        self.neighbor_sampling = neighbor_sampling

        # Contains all samples f(x) for each
        # point x in the form {x0: {0: f_0(x0), 1: f_1(x0), ...}, ...}
        self._data_samples = SortedDict()
        # Contains the number of samples taken
        # at each point x in the form {x0: n0, x1: n1, ...}
        self._number_samples = SortedDict()
        # This set contains the points x that have less than min_samples
        # samples or less than a (neighbor_sampling*100)% of their neighbors
        self._undersampled_points: set[Real] = set()
        # Contains the error in the estimate of the
        # mean at each point x in the form {x0: error(x0), ...}
        self.error: dict[Real, float] = decreasing_dict()
        #  Distance between two neighboring points in the
        # form {xi: ((xii-xi)^2 + (yii-yi)^2)^0.5, ...}
        self._distances: dict[Real, float] = decreasing_dict()
        # {xii: error[xii]/min(_distances[xi], _distances[xii], ...}
        self.rescaled_error: dict[Real, float] = decreasing_dict()

    @property
    def nsamples(self) -> int:
        """Returns the total number of samples"""
        return sum(self._number_samples.values())

    @property
    def min_samples_per_point(self) -> int:
        if not self._number_samples:
            return 0
        return min(self._number_samples.values())

    def ask(self, n: int, tell_pending: bool = True) -> tuple[Points, list[float]]:
        """Return 'n' points that are expected to maximally reduce the loss."""
        # If some point is undersampled, resample it
        if len(self._undersampled_points):
            x = next(iter(self._undersampled_points))
            points, loss_improvements = self._ask_for_more_samples(x, n)
        # If less than 2 points were sampled, sample a new one
        elif len(self.data) <= 1:
            # TODO: if `n` is very large, we should suggest a few different points.
            points, loss_improvements = self._ask_for_new_point(n)
        #  Else, check the resampling condition
        else:
            if len(self.rescaled_error):
                # This is in case rescaled_error is empty (e.g. when sigma=0)
                x, resc_error = self.rescaled_error.peekitem(0)
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

    def _ask_for_more_samples(self, x: Real, n: int) -> tuple[Points, list[float]]:
        """When asking for n points, the learner returns n times an existing point
        to be resampled, since in general n << min_samples and this point will
        need to be resampled many more times"""
        n_existing = self._number_samples.get(x, 0)
        points = [(seed + n_existing, x) for seed in range(n)]
        xl, xr = self.neighbors_combined[x]
        loss_left = self.losses_combined.get((xl, x), float("inf"))
        loss_right = self.losses_combined.get((x, xr), float("inf"))
        loss = (loss_left + loss_right) / 2
        if math.isinf(loss):
            loss_improvement = float("inf")
        else:
            loss_improvement = loss - loss * np.sqrt(n_existing) / np.sqrt(
                n_existing + n
            )
        loss_improvements = [loss_improvement / n] * n
        return points, loss_improvements

    def _ask_for_new_point(self, n: int) -> tuple[Points, list[float]]:
        """When asking for n new points, the learner returns n times a single
        new point, since in general n << min_samples and this point will need
        to be resampled many more times"""
        points, (loss_improvement,) = self._ask_points_without_adding(1)
        points = [(seed, x) for seed, x in zip(range(n), n * points)]
        loss_improvements = [loss_improvement / n] * n
        return points, loss_improvements

    def tell_pending(self, seed_x: Point) -> None:
        _, x = seed_x
        self.pending_points.add(seed_x)
        if x not in self.data:
            self._update_neighbors(x, self.neighbors_combined)
            self._update_losses(x, real=False)

    def tell(self, seed_x: Point, y: Real) -> None:
        seed, x = seed_x
        if y is None:
            raise TypeError(
                "Y-value may not be None, use learner.tell_pending(x)"
                "to indicate that this value is currently being calculated"
            )

        if x not in self.data:
            self._update_data(x, y, "new")
            self._update_data_structures(seed_x, y, "new")
        elif seed not in self._data_samples[x]:  # check if the seed is new
            self._update_data(x, y, "resampled")
            self._update_data_structures(seed_x, y, "resampled")
        self.pending_points.discard(seed_x)

    def _update_rescaled_error_in_mean(self, x: Real, point_type: str) -> None:
        """Updates ``self.rescaled_error``.

        Parameters
        ----------
        point_type : str
            Must be either "new" or "resampled".
        """
        #  Update neighbors
        x_left, x_right = self.neighbors[x]
        dists = self._distances
        if x_left is None and x_right is None:
            return

        if x_left is None:
            d_left = dists[x]
        else:
            d_left = dists[x_left]
            if x_left in self.rescaled_error:
                xll = self.neighbors[x_left][0]
                norm = dists[x_left] if xll is None else min(dists[xll], dists[x_left])
                self.rescaled_error[x_left] = self.error[x_left] / norm

        if x_right is None:
            d_right = dists[x_left]
        else:
            d_right = dists[x]
            if x_right in self.rescaled_error:
                xrr = self.neighbors[x_right][1]
                norm = dists[x] if xrr is None else min(dists[x], dists[x_right])
                self.rescaled_error[x_right] = self.error[x_right] / norm

        # Update x
        if point_type == "resampled":
            norm = min(d_left, d_right)
            self.rescaled_error[x] = self.error[x] / norm

    def _update_data(self, x: Real, y: Real, point_type: str) -> None:
        if point_type == "new":
            self.data[x] = y
        elif point_type == "resampled":
            n = len(self._data_samples[x])
            new_average = self.data[x] * n / (n + 1) + y / (n + 1)
            self.data[x] = new_average

    def _update_data_structures(self, seed_x: Point, y: Real, point_type: str) -> None:
        seed, x = seed_x
        if point_type == "new":
            self._data_samples[x] = {seed: y}

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
            self.error[x] = np.inf
            self.rescaled_error[x] = np.inf
            self._update_distances(x)
            self._update_rescaled_error_in_mean(x, "new")

        elif point_type == "resampled":
            self._data_samples[x][seed] = y
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
            ys = self._data_samples[x].values()
            self.error[x] = self._calc_error_in_mean(ys, y_avg, n)
            self._update_distances(x)
            self._update_rescaled_error_in_mean(x, "resampled")

            if self.error[x] <= self.min_error or n >= self.max_samples:
                self.rescaled_error.pop(x, None)

            # We also need to update scale and losses
            self._update_scale(x, y)
            self._update_losses_resampling(x, real=True)

            # If the scale has increased enough, recompute all losses.
            # We only update the scale considering resampled points, since new
            # points are more likely to be outliers.
            if self._scale[1] > self._recompute_losses_factor * self._oldscale[1]:
                for interval in reversed(self.losses):
                    self._update_interpolated_loss_in_interval(*interval)
                self._oldscale = deepcopy(self._scale)

    def _update_distances(self, x: Real) -> None:
        x_left, x_right = self.neighbors[x]
        y = self.data[x]
        if x_left is not None:
            self._distances[x_left] = hypot((x - x_left), (y - self.data[x_left]))
        if x_right is not None:
            self._distances[x] = hypot((x_right - x), (self.data[x_right] - y))

    def _update_losses_resampling(self, x: Real, real=True) -> None:
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

    def _calc_error_in_mean(self, ys: Iterable[Real], y_avg: Real, n: int) -> float:
        variance_in_mean = sum((y - y_avg) ** 2 for y in ys) / (n - 1)
        t_student = scipy.stats.t.ppf(1 - self.alpha, df=n - 1)
        return t_student * (variance_in_mean / n) ** 0.5

    def tell_many(self, xs: Points, ys: Sequence[Real]) -> None:
        # Check that all x are within the bounds
        # TODO: remove this requirement, all other learners add the data
        # but ignore it going forward.
        if not np.prod([x >= self.bounds[0] and x <= self.bounds[1] for _, x in xs]):
            raise ValueError(
                "x value out of bounds, "
                "remove x or enlarge the bounds of the learner"
            )

        # Create a mapping of points to a list of samples
        mapping: DefaultDict[Real, DefaultDict[int, Real]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for (seed, x), y in zip(xs, ys):
            mapping[x][seed] = y

        for x, seed_y_mapping in mapping.items():
            if len(seed_y_mapping) == 1:
                seed, y = list(seed_y_mapping.items())[0]
                self.tell((seed, x), y)
            elif len(seed_y_mapping) > 1:
                # If we stored more than 1 y-value for the previous x,
                # use a more efficient routine to tell many samples
                # simultaneously, before we move on to a new x
                self.tell_many_at_point(x, seed_y_mapping)

    def tell_many_at_point(self, x: Real, seed_y_mapping: dict[int, Real]) -> None:
        """Tell the learner about many samples at a certain location x.

        Parameters
        ----------
        x : float
            Value from the function domain.
        seed_y_mapping : Dict[int, Real]
            Dictionary of ``seed`` -> ``y`` at ``x``.
        """
        # Check x is within the bounds
        if not np.prod(x >= self.bounds[0] and x <= self.bounds[1]):
            raise ValueError(
                "x value out of bounds, "
                "remove x or enlarge the bounds of the learner"
            )

        # If x is a new point:
        if x not in self.data:
            # we make a copy because we don't want to modify the original dict
            seed_y_mapping = seed_y_mapping.copy()
            seed = next(iter(seed_y_mapping))
            y = seed_y_mapping.pop(seed)
            self._update_data(x, y, "new")
            self._update_data_structures((seed, x), y, "new")

        ys = np.array(list(seed_y_mapping.values()))

        # If x is not a new point or if there were more than 1 sample in ys:
        if len(ys) > 0:
            self._data_samples[x].update(seed_y_mapping)
            n = len(ys) + self._number_samples[x]
            self.data[x] = (
                np.mean(ys) * len(ys) + self.data[x] * self._number_samples[x]
            ) / n
            self._number_samples[x] = n
            # `self._update_data(x, y, "new")` included the point
            # in _undersampled_points. We remove it if there are
            # more than min_samples samples, disregarding neighbor_sampling.
            if n > self.min_samples:
                self._undersampled_points.discard(x)
            self.error[x] = self._calc_error_in_mean(
                self._data_samples[x].values(), self.data[x], n
            )
            self._update_distances(x)
            self._update_rescaled_error_in_mean(x, "resampled")
            if self.error[x] <= self.min_error or n >= self.max_samples:
                self.rescaled_error.pop(x, None)
            self._update_scale(x, min(self._data_samples[x].values()))
            self._update_scale(x, max(self._data_samples[x].values()))
            self._update_losses_resampling(x, real=True)
            if self._scale[1] > self._recompute_losses_factor * self._oldscale[1]:
                for interval in reversed(self.losses):
                    self._update_interpolated_loss_in_interval(*interval)
                self._oldscale = deepcopy(self._scale)

    def _get_data(self) -> dict[Real, Real]:
        return self._data_samples

    def _set_data(self, data: dict[Real, Real]) -> None:
        if data:
            for x, samples in data.items():
                self.tell_many_at_point(x, samples)

    def plot(self):
        """Returns a plot of the evaluated data with error bars (not implemented
           for vector functions, i.e., it requires vdim=1).

        Returns
        -------
        plot : `holoviews.element.Scatter * holoviews.element.ErrorBars *
                holoviews.element.Path`
            Plot of the evaluated data.
        """
        hv = ensure_holoviews()
        if not self.data:
            p = hv.Scatter([]) * hv.ErrorBars([]) * hv.Path([])
        elif not self.vdim > 1:
            xs, ys = zip(*sorted(self.data.items()))
            scatter = hv.Scatter(self.data)
            error = hv.ErrorBars([(x, self.data[x], self.error[x]) for x in self.data])
            line = hv.Path((xs, ys))
            p = scatter * error * line
        else:
            raise Exception("plot() not implemented for vector functions.")

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (self.bounds[1] - self.bounds[0])
        plot_bounds = (self.bounds[0] - margin, self.bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))


def decreasing_dict() -> dict:
    """This initialization orders the dictionary from large to small values"""

    def sorting_rule(key, value):
        return -value

    return ItemSortedDict(sorting_rule, SortedDict())
