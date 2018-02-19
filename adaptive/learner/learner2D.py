# -*- coding: utf-8 -*-
from collections import OrderedDict
from copy import copy
import itertools
from math import sqrt

import numpy as np
from scipy import interpolate

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner


# Learner2D and helper functions.

def deviations(ip):
    values = ip.values / (ip.values.ptp(axis=0).max() or 1)
    gradients = interpolate.interpnd.estimate_gradients_2d_global(
        ip.tri, values, tol=1e-6)

    p = ip.tri.points[ip.tri.vertices]
    vs = values[ip.tri.vertices]
    gs = gradients[ip.tri.vertices]

    def deviation(p, v, g):
        dev = 0
        for j in range(3):
            vest = v[:, j, None] + ((p[:, :, :] - p[:, j, None, :]) *
                                    g[:, j, None, :]).sum(axis=-1)
            dev += abs(vest - v).max(axis=1)
        return dev

    n_levels = vs.shape[2]
    devs = [deviation(p, vs[:, :, i], gs[:, :, i]) for i in range(n_levels)]
    return devs


def areas(ip):
    p = ip.tri.points[ip.tri.vertices]
    q = p[:, :-1, :] - p[:, -1, None, :]
    areas = abs(q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0]) / 2
    return areas


def uniform_sampling_2d(ip):
    """Loss function that samples the domain uniformly.

    Works with `~adaptive.Learner2D` only.

    Examples
    --------
    >>> def f(xy):
    ...     x, y = xy
    ...     return x**2 + y**2
    >>>
    >>> learner = adaptive.Learner2D(f,
    ...                              bounds=[(-1, -1), (1, 1)],
    ...                              loss_per_triangle=uniform_sampling_2d)
    >>>
    """
    return np.sqrt(areas(ip))


def _default_loss_per_triangle(ip):
    devs = deviations(ip)
    A = areas(ip)
    losses = np.sum([dev * np.sqrt(A) + 0.1 * A for dev in devs], axis=0)
    return losses


def choose_point_in_triangle(triangle, max_badness):
    """Choose a new point in inside a triangle.

    If the ratio of the longest edge of the triangle squared
    over the area is bigger than the `max_badness` the new point
    is chosen on the middle of the longest edge. Otherwise
    a point in the center of the triangle is chosen. The badness
    is 1 for a equilateral triangle.

    Parameters
    ----------
    triangle : numpy array
        The coordinates of a triangle with shape (3, 2)
    max_badness : int
        The badness at which the point is either chosen on a edge or
        in the middle.

    Returns
    -------
    point : numpy array
        The x and y coordinate of the suggested new point.
    """
    a, b, c = triangle
    area = 0.5 * np.cross(b - a, c - a)
    triangle_roll = np.roll(triangle, 1, axis=0)
    edge_lengths = np.linalg.norm(triangle - triangle_roll, axis=1)
    i = edge_lengths.argmax()

    # We multiply by sqrt(3) / 4 such that a equilateral triangle has badness=1
    badness = (edge_lengths[i]**2 / area) * (sqrt(3) / 4)
    if badness > max_badness:
        point = (triangle_roll[i] + triangle[i]) / 2
    else:
        point = triangle.mean(axis=0)
    return point


class Learner2D(BaseLearner):
    """Learns and predicts a function 'f: ℝ^2 → ℝ^N'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a tuple of two real
        parameters and return a real number.
    bounds : list of 2-tuples
        A list ``[(a1, b1), (a2, b2)]`` containing bounds,
        one per dimension.
    loss_per_triangle : callable, optional
        A function that returns the loss for every triangle.
        If not provided, then a default is used, which uses
        the deviation from a linear estimate, as well as
        triangle area, to determine the loss. See the notes
        for more details.


    Attributes
    ----------
    data : dict
        Sampled points and values.
    stack_size : int, default 10
        The size of the new candidate points stack. Set it to 1
        to recalculate the best points at each call to `choose_points`.

    Methods
    -------
    data_combined : dict
        Sampled points and values so far including
        the unknown interpolated ones.

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

    'loss_per_triangle' takes a single parameter, 'ip', which is a
    `scipy.interpolate.LinearNDInterpolator`. You can use the
    *undocumented* attributes 'tri' and 'values' of 'ip' to get a
    `scipy.spatial.Delaunay` and a vector of function values.
    These can be used to compute the loss. The functions
    `adaptive.learner.learner2D.areas` and
    `adaptive.learner.learner2D.deviations` to calculate the
    areas and deviations from a linear interpolation
    over each triangle.
    """

    def __init__(self, function, bounds, loss_per_triangle=None):
        self.ndim = len(bounds)
        self._vdim = None
        self.loss_per_triangle = loss_per_triangle or _default_loss_per_triangle
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self.data = OrderedDict()
        self._stack = OrderedDict()
        self._interp = set()

        xy_mean = np.mean(self.bounds, axis=1)
        xy_scale = np.ptp(self.bounds, axis=1)

        def scale(points):
            points = np.asarray(points)
            return (points - xy_mean) / xy_scale

        def unscale(points):
            points = np.asarray(points)
            return points * xy_scale + xy_mean

        self.scale = scale
        self.unscale = unscale

        self._bounds_points = list(itertools.product(*bounds))
        self._stack.update({p: np.inf for p in self._bounds_points})
        self.function = function
        self._ip = self._ip_combined = None
        self._loss = np.inf

        self.stack_size = 10

    @property
    def npoints(self):
        return len(self.data)

    @property
    def vdim(self):
        if self._vdim is None and self.data:
            try:
                value = next(iter(self.data.values()))
                self._vdim = len(value)
            except TypeError:
                self._vdim = 1
        return self._vdim if self._vdim is not None else 1

    @property
    def bounds_are_done(self):
        return not any((p in self._interp or p in self._stack)
                       for p in self._bounds_points)

    def data_combined(self):
        # Interpolate the unfinished points
        data_combined = copy(self.data)
        if self._interp:
            points_interp = list(self._interp)
            if self.bounds_are_done:
                values_interp = self.ip()(self.scale(points_interp))
            else:
                # Without the bounds the interpolation cannot be done properly,
                # so we just set everything to zero.
                values_interp = np.zeros((len(points_interp), self.vdim))

            for point, value in zip(points_interp, values_interp):
                data_combined[point] = value

        return data_combined

    def ip(self):
        if self._ip is None:
            points = self.scale(list(self.data.keys()))
            values = list(self.data.values())
            self._ip = interpolate.LinearNDInterpolator(points, values)
        return self._ip

    def ip_combined(self):
        if self._ip_combined is None:
            data_combined = self.data_combined()
            points = self.scale(list(data_combined.keys()))
            values = list(data_combined.values())
            self._ip_combined = interpolate.LinearNDInterpolator(points,
                                                                 values)
        return self._ip_combined

    def add_point(self, point, value):
        point = tuple(point)

        if value is None:
            self._interp.add(point)
            self._ip_combined = None
        else:
            self.data[point] = value
            self._interp.discard(point)
            self._ip = None

        self._stack.pop(point, None)

    def _fill_stack(self, stack_till=1):
        if len(self.data) + len(self._interp) < self.ndim + 1:
            raise ValueError("too few points...")

        # Interpolate
        ip = self.ip_combined()

        losses = self.loss_per_triangle(ip)

        points_new = []
        losses_new = []
        for j, _ in enumerate(losses):
            jsimplex = np.argmax(losses)
            triangle = ip.tri.points[ip.tri.vertices[jsimplex]]
            point_new = choose_point_in_triangle(triangle, max_badness=5)
            point_new = tuple(self.unscale(point_new))
            loss_new = losses[jsimplex]

            points_new.append(point_new)
            losses_new.append(loss_new)

            self._stack[point_new] = loss_new

            if len(self._stack) >= stack_till:
                break
            else:
                losses[jsimplex] = -np.inf

        return points_new, losses_new

    def choose_points(self, n, add_data=True):
        # Even if add_data is False we add the point such that _fill_stack
        # will return new points, later we remove these points if needed.
        points = list(self._stack.keys())
        loss_improvements = list(self._stack.values())
        n_left = n - len(points)
        self.add_data(points[:n], itertools.repeat(None))

        while n_left > 0:
            # The while loop is needed because `stack_till` could be larger
            # than the number of triangles between the points. Therefore
            # it could fill up till a length smaller than `stack_till`.
            new_points, new_loss_improvements = self._fill_stack(
                stack_till=max(n_left, self.stack_size))
            self.add_data(new_points[:n_left], itertools.repeat(None))
            n_left -= len(new_points)

            points += new_points
            loss_improvements += new_loss_improvements

        if not add_data:
            self._stack = OrderedDict(zip(points[:self.stack_size],
                                          loss_improvements))
            for point in points[:n]:
                self._interp.discard(point)

        return points[:n], loss_improvements[:n]

    def loss(self, real=True):
        if not self.bounds_are_done:
            return np.inf
        ip = self.ip() if real else self.ip_combined()
        losses = self.loss_per_triangle(ip)
        self._loss = losses.max()
        return self._loss

    def remove_unfinished(self):
        self._interp = set()

    def plot(self, n=None, tri_alpha=0):
        hv = ensure_holoviews()
        if self.vdim > 1:
            raise NotImplemented('holoviews currently does not support',
                                 '3D surface plots in bokeh.')
        x, y = self.bounds
        lbrt = x[0], y[0], x[1], y[1]

        if len(self.data) >= 4:
            ip = self.ip()

            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                n = int(0.658 / sqrt(areas(ip).min()))
                n = max(n, 10)

            x = y = np.linspace(-0.5, 0.5, n)
            z = ip(x[:, None], y[None, :]).squeeze()

            im = hv.Image(np.rot90(z), bounds=lbrt)

            if tri_alpha:
                points = self.unscale(ip.tri.points[ip.tri.vertices])
                points = np.pad(points[:, [0, 1, 2, 0], :],
                                pad_width=((0, 0), (0, 1), (0, 0)),
                                mode='constant',
                                constant_values=np.nan).reshape(-1, 2)
                tris = hv.EdgePaths([points])
            else:
                tris = hv.EdgePaths([])
        else:
            im = hv.Image([], bounds=lbrt)
            tris = hv.EdgePaths([])

        im_opts = dict(cmap='viridis')
        tri_opts = dict(line_width=0.5, alpha=tri_alpha)
        no_hover = dict(plot=dict(inspection_policy=None, tools=[]))

        return im.opts(style=im_opts) * tris.opts(style=tri_opts, **no_hover)
