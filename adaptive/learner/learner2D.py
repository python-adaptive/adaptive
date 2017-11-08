# -*- coding: utf-8 -*-
import itertools

import holoviews as hv
import numpy as np
from scipy import interpolate, special

from .base_learner import BaseLearner
from .utils import restore


# Learner2D and helper functions.

def _losses_per_triangle(ip):
    tri = ip.tri
    vs = ip.values.ravel()

    gradients = interpolate.interpnd.estimate_gradients_2d_global(
        tri, vs, tol=1e-6)
    p = tri.points[tri.vertices]
    g = gradients[tri.vertices]
    v = vs[tri.vertices]
    n_points_per_triangle = p.shape[1]

    dev = 0
    for j in range(n_points_per_triangle):
        vest = v[:, j, None] + ((p[:, :, :] - p[:, j, None, :]) *
                                g[:, j, None, :]).sum(axis=-1)
        dev += abs(vest - v).max(axis=1)

    q = p[:, :-1, :] - p[:, -1, None, :]
    areas = abs(q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0])
    areas /= special.gamma(n_points_per_triangle)
    areas = np.sqrt(areas)

    vs_scale = vs[tri.vertices].ptp()
    if vs_scale != 0:
        dev /= vs_scale

    return dev * areas

class Learner2D(BaseLearner):
    """Learns and predicts a function 'f: ℝ^2 → ℝ'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a tuple of two real
        parameters and return a real number.
    bounds : list of 2-tuples
        A list ``[(a1, b1), (a2, b2)]`` containing bounds,
        one per dimension.

    Attributes
    ----------
    points_combined
        Sample points so far including the unknown interpolated ones.
    values_combined
        Sampled values so far including the unknown interpolated ones.
    points
        Sample points so far with real results.
    values
        Sampled values so far with real results.

    Notes
    -----
    Adapted from an initial implementation by Pauli Virtanen.

    The sample points are chosen by estimating the point where the
    linear and cubic interpolants based on the existing points have
    maximal disagreement. This point is then taken as the next point
    to be sampled.

    In practice, this sampling protocol results to sparser sampling of
    smooth regions, and denser sampling of regions where the function
    changes rapidly, which is useful if the function is expensive to
    compute.

    This sampling procedure is not extremely fast, so to benefit from
    it, your function needs to be slow enough to compute.
    """

    def __init__(self, function, bounds):
        self.ndim = len(bounds)
        if self.ndim != 2:
            raise ValueError("Only 2-D sampling supported.")
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self._points = np.zeros([100, self.ndim])
        self._values = np.zeros([100], dtype=float)
        self._stack = []
        self._interp = {}

        xy_mean = np.mean(self.bounds, axis=1)
        xy_scale = np.ptp(self.bounds, axis=1)

        def scale(points):
            return (points - xy_mean) / xy_scale

        def unscale(points):
            return points * xy_scale + xy_mean

        self.scale = scale
        self.unscale = unscale

        # Keeps track till which index _points and _values are filled
        self.n = 0

        self._bounds_points = list(itertools.product(*bounds))

        # Add the loss improvement to the bounds in the stack
        self._stack = [(*p, np.inf) for p in self._bounds_points]

        self.function = function

    @property
    def points_combined(self):
        return self._points[:self.n]

    @property
    def values_combined(self):
        return self._values[:self.n]

    @property
    def points(self):
        return np.delete(self.points_combined,
                         list(self._interp.values()), axis=0)

    @property
    def values(self):
        return np.delete(self.values_combined,
                         list(self._interp.values()), axis=0)

    def ip(self):
        points = self.scale(self.points)
        return interpolate.LinearNDInterpolator(points, self.values)

    @property
    def n_real(self):
        return self.n - len(self._interp)

    def ip_combined(self):
        points = self.scale(self.points_combined)
        values = self.values_combined

        # Interpolate the unfinished points
        if self._interp:
            n_interp = list(self._interp.values())
            bounds_are_done = not any(p in self._interp
                                      for p in self._bounds_points)
            if bounds_are_done:
                values[n_interp] = self.ip()(points[n_interp])
            else:
                # It is important not to return exact zeros because
                # otherwise the algo will try to add the same point
                # to the stack each time.
                values[n_interp] = np.random.rand(len(n_interp)) * 1e-15

        return interpolate.LinearNDInterpolator(points, values)

    def add_point(self, point, value):
        nmax = self.values_combined.shape[0]
        if self.n >= nmax:
            self._values = np.resize(self._values, [2*nmax + 10])
            self._points = np.resize(self._points, [2*nmax + 10, self.ndim])

        point = tuple(point)

        # When the point is not evaluated yet, add an entry to self._interp
        # that saves the point and index.
        if value is None:
            self._interp[point] = self.n
            old_point = False
        else:
            old_point = point in self._interp

        # If the point is new add it a new value to _points and _values,
        # otherwise get the index of the value that is being replaced.
        if old_point:
            n = self._interp.pop(point)
        else:
            n = self.n
            self.n += 1

        self._points[n] = point
        self._values[n] = value

        # Remove the point if in the stack.
        for i, (*_point, _) in enumerate(self._stack):
            if point == tuple(_point):
                self._stack.pop(i)
                break

    def _fill_stack(self, stack_till=None):
        if stack_till is None:
            stack_till = 1

        if self.values_combined.shape[0] < self.ndim + 1:
            raise ValueError("too few points...")

        # Interpolate
        ip = self.ip_combined()
        tri = ip.tri

        losses = _losses_per_triangle(ip)

        def point_exists(p):
            eps = np.finfo(float).eps * self.points_combined.ptp() * 100
            if abs(p - self.points_combined).sum(axis=1).min() < eps:
                return True
            if self._stack:
                _stack_points, _ = self._split_stack()
                if abs(p - np.asarray(_stack_points)).sum(axis=1).min() < eps:
                    return True
            return False

        for j, _ in enumerate(losses):
            # Estimate point of maximum curvature inside the simplex
            jsimplex = np.argmax(losses)
            p = tri.points[tri.vertices[jsimplex]]
            point_new = self.unscale(p.mean(axis=-2))

            # XXX: not sure whether this is necessary it was there
            # originally.
            point_new = np.clip(point_new, *zip(*self.bounds))

            # Check if it is really new
            if point_exists(point_new):
                losses[jsimplex] = 0
                continue

            # Add to stack
            self._stack.append((*point_new, losses[jsimplex]))

            if len(self._stack) >= stack_till:
                break
            else:
                losses[jsimplex] = 0

    def _split_stack(self, n=None):
        points = []
        loss_improvements = []
        for *point, loss_improvement in self._stack[:n]:
            points.append(tuple(point))
            loss_improvements.append(loss_improvement)
        return points, loss_improvements

    def _choose_and_add_points(self, n):
        if n <= len(self._stack):
            points, loss_improvements = self._split_stack(n)
            self.add_data(points, itertools.repeat(None))
        else:
            points = []
            loss_improvements = []
            n_left = n
            while n_left > 0:
                # The while loop is needed because `stack_till` could be larger
                # than the number of triangles between the points. Therefore
                # it could fill up till a length smaller than `stack_till`.
                if self.n >= 2**self.ndim:
                    # Only fill the stack if no more bounds left in _stack
                    self._fill_stack(stack_till=n_left)
                new_points, new_loss_improvements = self._split_stack(n_left)
                points += new_points
                loss_improvements += new_loss_improvements
                self.add_data(new_points, itertools.repeat(None))
                n_left -= len(new_points)

        return points, loss_improvements

    def choose_points(self, n, add_data=True):
        if not add_data:
            with restore(self):
                return self._choose_and_add_points(n)
        else:
            return self._choose_and_add_points(n)

    def loss(self, real=True):
        n = self.n_real if real else self.n
        bounds_are_not_done = any(p in self._interp
                                  for p in self._bounds_points)
        if n <= 4 or bounds_are_not_done:
            return np.inf
        ip = self.ip() if real else self.ip_combined()
        losses = _losses_per_triangle(ip)
        return losses.max()

    def remove_unfinished(self):
        self._points = self.points.copy()
        self._values = self.values.copy()
        self.n -= len(self._interp)
        self._interp = {}

    def plot(self, n_x=201, n_y=201, triangles_alpha=0):
        x, y = self.bounds
        lbrt = x[0], y[0], x[1], y[1]
        if self.n_real >= 4:
            x = np.linspace(-0.5, 0.5, n_x)
            y = np.linspace(-0.5, 0.5, n_y)
            ip = self.ip()
            z = ip(x[:, None], y[None, :])
            plot = hv.Image(np.rot90(z), bounds=lbrt)

            if triangles_alpha:
                tri_points = self.unscale(ip.tri.points[ip.tri.vertices])
                contours = hv.Contours([p for p in tri_points])
                contours = contours.opts(style=dict(alpha=triangles_alpha))

        else:
            plot = hv.Image(np.zeros((2,2)), bounds=lbrt) # XXX: Change to `[]` when https://github.com/ioam/holoviews/pull/2088 is merged
            contours = hv.Contours([])

        return plot * contours if triangles_alpha else plot

