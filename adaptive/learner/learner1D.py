import collections.abc
import itertools
import math
from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import cloudpickle
import numpy as np
from sortedcollections.recipes import ItemSortedDict
from sortedcontainers.sorteddict import SortedDict

from adaptive.learner.base_learner import BaseLearner, uses_nth_neighbors
from adaptive.learner.learnerND import volume
from adaptive.learner.triangulation import simplex_volume_in_embedding
from adaptive.notebook_integration import ensure_holoviews
from adaptive.types import Float, Int, Real
from adaptive.utils import cache_latest

# -- types --

# Commonly used types
Interval = Union[Tuple[float, float], Tuple[float, float, int]]
NeighborsType = Dict[float, List[Union[float, None]]]

# Types for loss_per_interval functions
NoneFloat = Union[Float, None]
NoneArray = Union[np.ndarray, None]
XsType0 = Tuple[Float, Float]
YsType0 = Union[Tuple[Float, Float], Tuple[np.ndarray, np.ndarray]]
XsType1 = Tuple[NoneFloat, NoneFloat, NoneFloat, NoneFloat]
YsType1 = Union[
    Tuple[NoneFloat, NoneFloat, NoneFloat, NoneFloat],
    Tuple[NoneArray, NoneArray, NoneArray, NoneArray],
]
XsTypeN = Tuple[NoneFloat, ...]
YsTypeN = Union[Tuple[NoneFloat, ...], Tuple[NoneArray, ...]]


__all__ = [
    "uniform_loss",
    "default_loss",
    "abs_min_log_loss",
    "triangle_loss",
    "resolution_loss_function",
    "curvature_loss_function",
    "Learner1D",
]


@uses_nth_neighbors(0)
def uniform_loss(xs: XsType0, ys: YsType0) -> Float:
    """Loss function that samples the domain uniformly.

    Works with `~adaptive.Learner1D` only.

    Examples
    --------
    >>> def f(x):
    ...     return x**2
    >>>
    >>> learner = adaptive.Learner1D(f,
    ...                              bounds=(-1, 1),
    ...                              loss_per_interval=uniform_sampling_1d)
    >>>
    """
    dx = xs[1] - xs[0]
    return dx


@uses_nth_neighbors(0)
def default_loss(xs: XsType0, ys: YsType0) -> Float:
    """Calculate loss on a single interval.

    Currently returns the rescaled length of the interval. If one of the
    y-values is missing, returns 0 (so the intervals with missing data are
    never touched. This behavior should be improved later.
    """
    dx = xs[1] - xs[0]
    if isinstance(ys[0], collections.abc.Iterable):
        dy_vec = np.array([abs(a - b) for a, b in zip(*ys)])
        return np.hypot(dx, dy_vec).max()
    else:
        dy = ys[1] - ys[0]
        return np.hypot(dx, dy)


@uses_nth_neighbors(0)
def abs_min_log_loss(xs: XsType0, ys: YsType0) -> Float:
    """Calculate loss of a single interval that prioritizes the absolute minimum."""
    ys = tuple(np.log(np.abs(y).min()) for y in ys)
    return default_loss(xs, ys)


@uses_nth_neighbors(1)
def triangle_loss(xs: XsType1, ys: YsType1) -> Float:
    assert len(xs) == 4
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]

    if len(xs) == 2:  # we do not have enough points for a triangle
        return xs[1] - xs[0]

    N = len(xs) - 2  # number of constructed triangles
    if isinstance(ys[0], collections.abc.Iterable):
        pts = [(x, *y) for x, y in zip(xs, ys)]
        vol = simplex_volume_in_embedding
    else:
        pts = [(x, y) for x, y in zip(xs, ys)]
        vol = volume
    return sum(vol(pts[i : i + 3]) for i in range(N)) / N


def resolution_loss_function(
    min_length: Real = 0, max_length: Real = 1
) -> Callable[[XsType0, YsType0], Float]:
    """Loss function that is similar to the `default_loss` function, but you
    can set the maximum and minimum size of an interval.

    Works with `~adaptive.Learner1D` only.

    The arguments `min_length` and `max_length` should be in between 0 and 1
    because the total size is normalized to 1.

    Returns
    -------
    loss_function : callable

    Examples
    --------
    >>> def f(x):
    ...     return x**2
    >>>
    >>> loss = resolution_loss_function(min_length=0.01, max_length=1)
    >>> learner = adaptive.Learner1D(f, bounds=(-1, -1), loss_per_interval=loss)
    """

    @uses_nth_neighbors(0)
    def resolution_loss(xs: XsType0, ys: YsType0) -> Float:
        loss = uniform_loss(xs, ys)
        if loss < min_length:
            # Return zero such that this interval won't be chosen again
            return 0
        if loss > max_length:
            # Return infinite such that this interval will be picked
            return np.inf
        loss = default_loss(xs, ys)
        return loss

    return resolution_loss


def curvature_loss_function(
    area_factor: Real = 1, euclid_factor: Real = 0.02, horizontal_factor: Real = 0.02
) -> Callable[[XsType1, YsType1], Float]:
    # XXX: add a doc-string
    @uses_nth_neighbors(1)
    def curvature_loss(xs: XsType1, ys: YsType1) -> Float:
        xs_middle = xs[1:3]
        ys_middle = ys[1:3]

        triangle_loss_ = triangle_loss(xs, ys)
        default_loss_ = default_loss(xs_middle, ys_middle)
        dx = xs_middle[1] - xs_middle[0]
        return (
            area_factor * (triangle_loss_**0.5)
            + euclid_factor * default_loss_
            + horizontal_factor * dx
        )

    return curvature_loss


def linspace(x_left: Real, x_right: Real, n: Int) -> List[Float]:
    """This is equivalent to
    'np.linspace(x_left, x_right, n, endpoint=False)[1:]',
    but it is 15-30 times faster for small 'n'."""
    if n == 1:
        # This is just an optimization
        return []
    else:
        step = (x_right - x_left) / n
        return [x_left + step * i for i in range(1, n)]


def _get_neighbors_from_array(xs: np.ndarray) -> NeighborsType:
    xs = np.sort(xs)
    xs_left = np.roll(xs, 1).tolist()
    xs_right = np.roll(xs, -1).tolist()
    xs_left[0] = None
    xs_right[-1] = None
    neighbors = {x: [x_L, x_R] for x, x_L, x_R in zip(xs, xs_left, xs_right)}
    return SortedDict(neighbors)


def _get_intervals(
    x: float, neighbors: NeighborsType, nth_neighbors: int
) -> List[Tuple[float, float]]:
    nn = nth_neighbors
    i = neighbors.index(x)
    start = max(0, i - nn - 1)
    end = min(len(neighbors), i + nn + 2)
    points = neighbors.keys()[start:end]
    return list(zip(points, points[1:]))


class Learner1D(BaseLearner):
    """Learns and predicts a function 'f:ℝ → ℝ^N'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single real parameter and
        return a real number or 1D array.
    bounds : pair of reals
        The bounds of the interval on which to learn 'function'.
    loss_per_interval: callable, optional
        A function that returns the loss for a single interval of the domain.
        If not provided, then a default is used, which uses the scaled distance
        in the x-y plane as the loss. See the notes for more details.

    Attributes
    ----------
    data : dict
        Sampled points and values.
    pending_points : set
        Points that still have to be evaluated.

    Notes
    -----
    `loss_per_interval` takes 2 parameters: ``xs`` and ``ys``, and returns a
        scalar; the loss over the interval.
    xs : tuple of floats
        The x values of the interval, if `nth_neighbors` is greater than zero it
        also contains the x-values of the neighbors of the interval, in ascending
        order. The interval we want to know the loss of is then the middle
        interval. If no neighbor is available (at the edges of the domain) then
        `None` will take the place of the x-value of the neighbor.
    ys : tuple of function values
        The output values of the function when evaluated at the `xs`. This is
        either a float or a tuple of floats in the case of vector output.


    The `loss_per_interval` function may also have an attribute `nth_neighbors`
    that indicates how many of the neighboring intervals to `interval` are used.
    If `loss_per_interval` doesn't  have such an attribute, it's assumed that is
    uses **no** neighboring intervals. Also see the `uses_nth_neighbors`
    decorator for more information.
    """

    def __init__(
        self,
        function: Callable[[Real], Union[Float, np.ndarray]],
        bounds: Tuple[Real, Real],
        loss_per_interval: Optional[Callable[[XsTypeN, YsTypeN], Float]] = None,
    ):
        self.function = function  # type: ignore

        if hasattr(loss_per_interval, "nth_neighbors"):
            self.nth_neighbors = loss_per_interval.nth_neighbors
        else:
            self.nth_neighbors = 0

        self.loss_per_interval = loss_per_interval or default_loss

        # When the scale changes by a factor 2, the losses are
        # recomputed. This is tunable such that we can test
        # the learners behavior in the tests.
        self._recompute_losses_factor = 2

        self.data: Dict[Real, Real] = {}
        self.pending_points: Set[Real] = set()

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self.neighbors: NeighborsType = SortedDict()
        self.neighbors_combined: NeighborsType = SortedDict()

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [list(bounds), [np.inf, -np.inf]]

        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [bounds[1] - bounds[0], 0]
        self._oldscale = deepcopy(self._scale)

        # A LossManager storing the loss function for each interval x_n.
        self.losses = loss_manager(self._scale[0])
        self.losses_combined = loss_manager(self._scale[0])

        # The precision in 'x' below which we set losses to 0.
        self._dx_eps = 2 * max(np.abs(bounds)) * np.finfo(float).eps

        self.bounds = list(bounds)
        self.__missing_bounds = set(self.bounds)  # cache of missing bounds

        self._vdim: Optional[int] = None

    @property
    def vdim(self) -> int:
        """Length of the output of ``learner.function``.
        If the output is unsized (when it's a scalar)
        then `vdim = 1`.

        As long as no data is known `vdim = 1`.
        """
        if self._vdim is None:
            if self.data:
                y = next(iter(self.data.values()))
                try:
                    self._vdim = len(np.squeeze(y))
                except TypeError:
                    # Means we are taking the length of a float
                    self._vdim = 1
            else:
                return 1
        return self._vdim

    def to_numpy(self):
        """Data as NumPy array of size ``(npoints, 2)`` if ``learner.function`` returns a scalar
        and ``(npoints, 1+vdim)`` if ``learner.function`` returns a vector of length ``vdim``."""
        return np.array([(x, *np.atleast_1d(y)) for x, y in sorted(self.data.items())])

    @property
    def npoints(self) -> int:
        """Number of evaluated points."""
        return len(self.data)

    @cache_latest
    def loss(self, real: bool = True) -> float:
        if self._missing_bounds():
            return np.inf
        losses = self.losses if real else self.losses_combined
        if not losses:
            return np.inf
        max_interval, max_loss = losses.peekitem(0)
        return max_loss

    def _scale_x(self, x: Optional[Float]) -> Optional[Float]:
        if x is None:
            return None
        return x / self._scale[0]

    def _scale_y(
        self, y: Union[Float, np.ndarray, None]
    ) -> Union[Float, np.ndarray, None]:
        if y is None:
            return None
        y_scale = self._scale[1] or 1
        return y / y_scale

    def _get_point_by_index(self, ind: int) -> Optional[float]:
        if ind < 0 or ind >= len(self.neighbors):
            return None
        return self.neighbors.keys()[ind]

    def _get_loss_in_interval(self, x_left: float, x_right: float) -> float:
        assert x_left is not None and x_right is not None

        if x_right - x_left < self._dx_eps:
            return 0

        nn = self.nth_neighbors
        i = self.neighbors.index(x_left)
        start = i - nn
        end = i + nn + 2

        xs = [self._get_point_by_index(i) for i in range(start, end)]
        ys = [self.data.get(x, None) for x in xs]

        xs_scaled = tuple(self._scale_x(x) for x in xs)
        ys_scaled = tuple(self._scale_y(y) for y in ys)

        # we need to compute the loss for this interval
        return self.loss_per_interval(xs_scaled, ys_scaled)

    def _update_interpolated_loss_in_interval(
        self, x_left: float, x_right: float
    ) -> None:
        if x_left is None or x_right is None:
            return

        loss = self._get_loss_in_interval(x_left, x_right)
        self.losses[x_left, x_right] = loss

        # Iterate over all interpolated intervals in between
        # x_left and x_right and set the newly interpolated loss.
        a, b = x_left, None
        dx = x_right - x_left
        while b != x_right:
            b = self.neighbors_combined[a][1]
            self.losses_combined[a, b] = (b - a) * loss / dx
            a = b

    def _update_losses(self, x: float, real: bool = True) -> None:
        """Update all losses that depend on x"""
        # When we add a new point x, we should update the losses
        # (x_left, x_right) are the "real" neighbors of 'x'.
        x_left, x_right = self._find_neighbors(x, self.neighbors)
        # (a, b) are the neighbors of the combined interpolated
        # and "real" intervals.
        a, b = self._find_neighbors(x, self.neighbors_combined)

        # (a, b) is splitted into (a, x) and (x, b) so if (a, b) exists
        self.losses_combined.pop((a, b), None)  # we get rid of (a, b).

        if real:
            # We need to update all interpolated losses in the interval
            # (x_left, x), (x, x_right) and the nth_neighbors nearest
            # neighboring intervals. Since the addition of the
            # point 'x' could change their loss.
            for ival in _get_intervals(x, self.neighbors, self.nth_neighbors):
                self._update_interpolated_loss_in_interval(*ival)

            # Since 'x' is in between (x_left, x_right),
            # we get rid of the interval.
            self.losses.pop((x_left, x_right), None)
            self.losses_combined.pop((x_left, x_right), None)
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

    @staticmethod
    def _find_neighbors(x: float, neighbors: NeighborsType) -> Any:
        if x in neighbors:
            return neighbors[x]
        pos = neighbors.bisect_left(x)
        keys = neighbors.keys()
        x_left = keys[pos - 1] if pos != 0 else None
        x_right = keys[pos] if pos != len(neighbors) else None
        return x_left, x_right

    def _update_neighbors(self, x: float, neighbors: NeighborsType) -> None:
        if x not in neighbors:  # The point is new
            x_left, x_right = self._find_neighbors(x, neighbors)
            neighbors[x] = [x_left, x_right]
            neighbors.get(x_left, [None, None])[1] = x
            neighbors.get(x_right, [None, None])[0] = x

    def _update_scale(self, x: float, y: Union[Float, np.ndarray]) -> None:
        """Update the scale with which the x and y-values are scaled.

        For a learner where the function returns a single scalar the scale
        is determined by the peak-to-peak value of the x and y-values.

        When the function returns a vector the learners y-scale is set by
        the level with the the largest peak-to-peak value.
        """
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        self._scale[0] = self._bbox[0][1] - self._bbox[0][0]
        if y is not None:
            if self.vdim > 1:
                try:
                    y_min = np.nanmin([self._bbox[1][0], y], axis=0)
                    y_max = np.nanmax([self._bbox[1][1], y], axis=0)
                except ValueError:
                    # Happens when `_bbox[1]` is a float and `y` a vector.
                    y_min = y_max = y
                self._bbox[1] = [y_min, y_max]
                self._scale[1] = np.max(y_max - y_min)
            else:
                self._bbox[1][0] = min(self._bbox[1][0], y)
                self._bbox[1][1] = max(self._bbox[1][1], y)
                self._scale[1] = self._bbox[1][1] - self._bbox[1][0]

    def tell(self, x: float, y: Union[Float, Sequence[Float], np.ndarray]) -> None:
        if x in self.data:
            # The point is already evaluated before
            return
        if y is None:
            raise TypeError(
                "Y-value may not be None, use learner.tell_pending(x)"
                "to indicate that this value is currently being calculated"
            )

        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        # Add point to the real data dict
        self.data[x] = y

        # remove from set of pending points
        self.pending_points.discard(x)

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

    def tell_pending(self, x: float) -> None:
        if x in self.data:
            # The point is already evaluated before
            return
        self.pending_points.add(x)
        self._update_neighbors(x, self.neighbors_combined)
        self._update_losses(x, real=False)

    def tell_many(
        self,
        xs: Sequence[Float],
        ys: Union[
            Sequence[Float],
            Sequence[Sequence[Float]],
            Sequence[np.ndarray],
        ],
        *,
        force: bool = False
    ) -> None:
        if not force and not (len(xs) > 0.5 * len(self.data) and len(xs) > 2):
            # Only run this more efficient method if there are
            # at least 2 points and the amount of points added are
            # at least half of the number of points already in 'data'.
            # These "magic numbers" are somewhat arbitrary.
            super().tell_many(xs, ys)
            return

        # Add data points
        self.data.update(zip(xs, ys))
        self.pending_points.difference_update(xs)

        # Get all data as numpy arrays
        points = np.array(list(self.data.keys()))
        values = np.array(list(self.data.values()))
        points_pending = np.array(list(self.pending_points))
        points_combined = np.hstack([points_pending, points])

        # Generate neighbors
        self.neighbors = _get_neighbors_from_array(points)
        self.neighbors_combined = _get_neighbors_from_array(points_combined)

        # Update scale
        self._bbox[0] = [points_combined.min(), points_combined.max()]
        self._bbox[1] = [values.min(axis=0), values.max(axis=0)]
        self._scale[0] = self._bbox[0][1] - self._bbox[0][0]
        self._scale[1] = np.max(self._bbox[1][1] - self._bbox[1][0])
        self._oldscale = deepcopy(self._scale)

        # Find the intervals for which the losses should be calculated.
        intervals, intervals_combined = (
            [(x_m, x_r) for x_m, (x_l, x_r) in neighbors.items()][:-1]
            for neighbors in (self.neighbors, self.neighbors_combined)
        )

        # The the losses for the "real" intervals.
        self.losses = loss_manager(self._scale[0])
        for ival in intervals:
            self.losses[ival] = self._get_loss_in_interval(*ival)

        # List with "real" intervals that have interpolated intervals inside
        to_interpolate = []

        self.losses_combined = loss_manager(self._scale[0])
        for ival in intervals_combined:
            # If this interval exists in 'losses' then copy it otherwise
            # calculate it.
            if ival in reversed(self.losses):
                self.losses_combined[ival] = self.losses[ival]
            else:
                # Set all losses to inf now, later they might be udpdated if the
                # interval appears to be inside a real interval.
                self.losses_combined[ival] = np.inf
                x_left, x_right = ival
                a, b = to_interpolate[-1] if to_interpolate else (None, None)
                if b == x_left and (a, b) not in self.losses:
                    # join (a, b) and (x_left, x_right) → (a, x_right)
                    to_interpolate[-1] = (a, x_right)
                else:
                    to_interpolate.append((x_left, x_right))

        for ival in to_interpolate:
            if ival in reversed(self.losses):
                # If this interval does not exist it should already
                # have an inf loss.
                self._update_interpolated_loss_in_interval(*ival)

    def ask(self, n: int, tell_pending: bool = True) -> Tuple[List[float], List[float]]:
        """Return 'n' points that are expected to maximally reduce the loss."""
        points, loss_improvements = self._ask_points_without_adding(n)

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _missing_bounds(self) -> List[Real]:
        missing_bounds = []
        for b in copy(self.__missing_bounds):
            if b in self.data:
                self.__missing_bounds.remove(b)
            elif b not in self.pending_points:
                missing_bounds.append(b)
        return sorted(missing_bounds)

    def _ask_points_without_adding(self, n: int) -> Tuple[List[float], List[float]]:
        """Return 'n' points that are expected to maximally reduce the loss.
        Without altering the state of the learner"""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.
        # Return equally spaced points within each interval to which points
        # will be added.

        # XXX: when is this used and could we safely remove it without impacting performance?
        if n == 0:
            return [], []

        # If the bounds have not been chosen yet, we choose them first.
        missing_bounds = self._missing_bounds()
        if len(missing_bounds) >= n:
            return missing_bounds[:n], [np.inf] * n

        # Add bound intervals to quals if bounds were missing.
        if len(self.data) + len(self.pending_points) == 0:
            # We don't have any points, so return a linspace with 'n' points.
            a, b = self.bounds
            return np.linspace(a, b, n).tolist(), [np.inf] * n

        quals = loss_manager(self._scale[0])
        if len(missing_bounds) > 0:
            # There is at least one point in between the bounds.
            all_points = list(self.data.keys()) + list(self.pending_points)
            intervals = [
                (self.bounds[0], min(all_points)),
                (max(all_points), self.bounds[1]),
            ]
            for interval, bound in zip(intervals, self.bounds):
                if bound in missing_bounds:
                    quals[(*interval, 1)] = np.inf

        points_to_go = n - len(missing_bounds)

        # Calculate how many points belong to each interval.
        i, i_max = 0, len(self.losses_combined)
        for _ in range(points_to_go):
            qual, loss_qual = quals.peekitem(0) if quals else (None, 0)
            ival, loss_ival = (
                self.losses_combined.peekitem(i) if i < i_max else (None, 0)
            )

            if qual is None or (
                ival is not None
                and self._loss(self.losses_combined, ival) >= self._loss(quals, qual)
            ):
                i += 1
                quals[(*ival, 2)] = loss_ival / 2
            else:
                quals.pop(qual, None)
                *xs, n = qual
                quals[(*xs, n + 1)] = loss_qual * n / (n + 1)

        points = list(
            itertools.chain.from_iterable(linspace(*ival, n) for (*ival, n) in quals)
        )

        loss_improvements = list(
            itertools.chain.from_iterable(
                itertools.repeat(quals[x0, x1, n], n - 1) for (x0, x1, n) in quals
            )
        )

        # add the missing bounds
        points = missing_bounds + points
        loss_improvements = [np.inf] * len(missing_bounds) + loss_improvements

        return points, loss_improvements

    def _loss(
        self, mapping: Dict[Interval, float], ival: Interval
    ) -> Tuple[float, Interval]:
        loss = mapping[ival]
        return finite_loss(ival, loss, self._scale[0])

    def plot(self, *, scatter_or_line: str = "scatter"):
        """Returns a plot of the evaluated data.

        Parameters
        ----------
        scatter_or_line : str, default: "scatter"
            Plot as a scatter plot ("scatter") or a line plot ("line").

        Returns
        -------
        plot : `holoviews.Overlay`
            Plot of the evaluated data.
        """
        if scatter_or_line not in ("scatter", "line"):
            raise ValueError("scatter_or_line must be 'scatter' or 'line'")
        hv = ensure_holoviews()

        xs, ys = zip(*sorted(self.data.items())) if self.data else ([], [])
        if scatter_or_line == "scatter":
            if self.vdim == 1:
                plots = [hv.Scatter((xs, ys))]
            else:
                plots = [hv.Scatter((xs, _ys)) for _ys in np.transpose(ys)]
        else:
            plots = [hv.Path((xs, ys))]

        # Put all plots in an Overlay because a DynamicMap can't handle changing
        # datatypes, e.g. when `vdim` isn't yet known and the live_plot is running.
        p = hv.Overlay(plots)
        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (self.bounds[1] - self.bounds[0])
        plot_bounds = (self.bounds[0] - margin, self.bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))

    def remove_unfinished(self) -> None:
        self.pending_points = set()
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)

    def _get_data(self) -> Dict[float, float]:
        return self.data

    def _set_data(self, data: Dict[float, float]) -> None:
        if data:
            xs, ys = zip(*data.items())
            self.tell_many(xs, ys)

    def __getstate__(self):
        return (
            cloudpickle.dumps(self.function),
            tuple(self.bounds),
            self.loss_per_interval,
            dict(self.losses),  # SortedDict cannot be pickled
            dict(self.losses_combined),  # ItemSortedDict cannot be pickled
            self._get_data(),
        )

    def __setstate__(self, state):
        function, bounds, loss_per_interval, losses, losses_combined, data = state
        function = cloudpickle.loads(function)
        self.__init__(function, bounds, loss_per_interval)
        self._set_data(data)
        self.losses.update(losses)
        self.losses_combined.update(losses_combined)


def loss_manager(x_scale: float) -> Dict[Interval, float]:
    def sort_key(ival, loss):
        loss, ival = finite_loss(ival, loss, x_scale)
        return -loss, ival

    sorted_dict = ItemSortedDict(sort_key)
    return sorted_dict


def finite_loss(ival: Interval, loss: float, x_scale: float) -> Tuple[float, Interval]:
    """Get the so-called finite_loss of an interval in order to be able to
    sort intervals that have infinite loss."""
    # If the loss is infinite we return the
    # distance between the two points.
    if math.isinf(loss) or math.isnan(loss):
        loss = (ival[1] - ival[0]) / x_scale
        if len(ival) == 3:
            # Used when constructing quals. Last item is
            # the number of points inside the qual.
            loss /= ival[2]

    # We round the loss to 12 digits such that losses
    # are equal up to numerical precision will be considered
    # equal. This is 3.5x faster than unsing the `round` function.
    round_fac = 1e12
    loss = int(loss * round_fac + 0.5) / round_fac
    return loss, ival
