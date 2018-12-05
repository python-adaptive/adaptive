# -*- coding: utf-8 -*-

from collections import OrderedDict
from copy import copy
import itertools
from math import sqrt

import numpy as np
from scipy import interpolate

from adaptive.learner.base_learner import BaseLearner
from adaptive.notebook_integration import ensure_holoviews
from adaptive.utils import cache_latest


# Learner2D and helper functions.

def deviations(ip):
    """Returns the deviation of the linear estimate.

    Is useful when defining custom loss functions.

    Parameters
    ----------
    ip : `scipy.interpolate.LinearNDInterpolator` instance

    Returns
    -------
    numpy array
        The deviation per triangle.
    """
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
    """Returns the area per triangle of the triangulation inside
    a `LinearNDInterpolator` instance.

    Is useful when defining custom loss functions.

    Parameters
    ----------
    ip : `scipy.interpolate.LinearNDInterpolator` instance

    Returns
    -------
    numpy array
        The area per triangle in ``ip.tri``.
    """
    p = ip.tri.points[ip.tri.vertices]
    q = p[:, :-1, :] - p[:, -1, None, :]
    areas = abs(q[:, 0, 0] * q[:, 1, 1] - q[:, 0, 1] * q[:, 1, 0]) / 2
    return areas


def uniform_loss(ip):
    """Loss function that samples the domain uniformly.

    Works with `~adaptive.Learner2D` only.

    Examples
    --------
    >>> from adaptive.learner.learner2D import uniform_loss
    >>> def f(xy):
    ...     x, y = xy
    ...     return x**2 + y**2
    >>>
    >>> learner = adaptive.Learner2D(f,
    ...                              bounds=[(-1, -1), (1, 1)],
    ...                              loss_per_triangle=uniform_loss)
    >>>
    """
    return np.sqrt(areas(ip))


def resolution_loss_function(min_distance=0, max_distance=1):
    """Loss function that is similar to the `default_loss` function, but you
    can set the maximimum and minimum size of a triangle.

    Works with `~adaptive.Learner2D` only.

    The arguments `min_distance` and `max_distance` should be in between 0 and 1
    because the total area is normalized to 1.

    Examples
    --------
    >>> def f(xy):
    ...     x, y = xy
    ...     return x**2 + y**2
    >>>
    >>> loss = resolution_loss_function(min_distance=0.01, max_distance=1)
    >>> learner = adaptive.Learner2D(f,
    ...                              bounds=[(-1, -1), (1, 1)],
    ...                              loss_per_triangle=loss)
    >>>
    """
    def resolution_loss(ip):
        loss = default_loss(ip)

        A = areas(ip)
        # Setting areas with a small area to zero such that they won't be chosen again
        loss[A < min_distance**2] = 0

        # Setting triangles that have a size larger than max_distance to infinite loss
        # such that these triangles will be picked
        loss[A > max_distance**2] = np.inf

        return loss
    return resolution_loss


def minimize_triangle_surface_loss(ip):
    """Loss function that is similar to the default loss function in the
    `~adaptive.Learner1D`. The loss is the area spanned by the 3D
    vectors of the vertices.

    Works with `~adaptive.Learner2D` only.

    Examples
    --------
    >>> from adaptive.learner.learner2D import minimize_triangle_surface_loss
    >>> def f(xy):
    ...     x, y = xy
    ...     return x**2 + y**2
    >>>
    >>> learner = adaptive.Learner2D(f, bounds=[(-1, -1), (1, 1)],
    ...     loss_per_triangle=minimize_triangle_surface_loss)
    >>>
    """
    tri = ip.tri
    points = tri.points[tri.vertices]
    values = ip.values[tri.vertices]
    values = values / (ip.values.ptp(axis=0).max() or 1)

    def _get_vectors(points):
        delta = points - points[:, -1, :][:, None, :]
        vectors = delta[:, :2, :]
        return vectors[:, 0, :], vectors[:, 1, :]

    a_xy, b_xy = _get_vectors(points)
    a_z, b_z = _get_vectors(values)

    a = np.hstack([a_xy, a_z])
    b = np.hstack([b_xy, b_z])

    return np.linalg.norm(np.cross(a, b) / 2, axis=1)


def default_loss(ip):
    dev = np.sum(deviations(ip), axis=0)
    A = areas(ip)
    losses = dev * np.sqrt(A) + 0.3 * A
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
    pending_points : set
        Points that still have to be evaluated and are currently
        interpolated, see `data_combined`.
    stack_size : int, default: 10
        The size of the new candidate points stack. Set it to 1
        to recalculate the best points at each call to `ask`.
    aspect_ratio : float, int, default: 1
        Average ratio of ``x`` span over ``y`` span of a triangle. If
        there is more detail in either ``x`` or ``y`` the ``aspect_ratio``
        needs to be adjusted. When ``aspect_ratio > 1`` the
        triangles will be stretched along ``x``, otherwise
        along ``y``.

    Methods
    -------
    data_combined : dict
        Sampled points and values so far including
        the unknown interpolated points in `pending_points`.

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

    `loss_per_triangle` takes a single parameter, `ip`, which is a
    `scipy.interpolate.LinearNDInterpolator`. You can use the
    *undocumented* attributes ``tri`` and ``values`` of `ip` to get a
    `scipy.spatial.Delaunay` and a vector of function values.
    These can be used to compute the loss. The functions
    `~adaptive.learner.learner2D.areas` and
    `~adaptive.learner.learner2D.deviations` to calculate the
    areas and deviations from a linear interpolation
    over each triangle.
    """

    def __init__(self, function, bounds, loss_per_triangle=None):
        self.ndim = len(bounds)
        self._vdim = None
        self.loss_per_triangle = loss_per_triangle or default_loss
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self.data = OrderedDict()
        self._stack = OrderedDict()
        self.pending_points = set()

        self.xy_mean = np.mean(self.bounds, axis=1)
        self._xy_scale = np.ptp(self.bounds, axis=1)
        self.aspect_ratio = 1

        self._bounds_points = list(itertools.product(*bounds))
        self._stack.update({p: np.inf for p in self._bounds_points})
        self.function = function
        self._ip = self._ip_combined = None

        self.stack_size = 10

    @property
    def xy_scale(self):
        xy_scale = self._xy_scale
        if self.aspect_ratio == 1:
            return xy_scale
        else:
            return np.array([xy_scale[0], xy_scale[1] / self.aspect_ratio])

    def _scale(self, points):
        points = np.asarray(points, dtype=float)
        return (points - self.xy_mean) / self.xy_scale

    def _unscale(self, points):
        points = np.asarray(points, dtype=float)
        return points * self.xy_scale + self.xy_mean

    @property
    def npoints(self):
        """Number of evaluated points."""
        return len(self.data)

    @property
    def vdim(self):
        """Length of the output of ``learner.function``.
        If the output is unsized (when it's a scalar)
        then `vdim = 1`.

        As long as no data is known `vdim = 1`.
        """
        if self._vdim is None and self.data:
            try:
                value = next(iter(self.data.values()))
                self._vdim = len(value)
            except TypeError:
                self._vdim = 1
        return self._vdim or 1

    @property
    def bounds_are_done(self):
        return not any((p in self.pending_points or p in self._stack)
                       for p in self._bounds_points)

    def _data_in_bounds(self):
        if self.data:
            points = np.array(list(self.data.keys()))
            values = np.array(list(self.data.values()), dtype=float)
            ll, ur = np.reshape(self.bounds, (2, 2)).T
            inds = np.all(np.logical_and(ll <= points, points <= ur), axis=1)
            return points[inds], values[inds].reshape(-1, self.vdim)
        return np.zeros((0, 2)), np.zeros((0, self.vdim), dtype=float)

    def _data_interp(self):
        if self.pending_points:
            points = list(self.pending_points)
            if self.bounds_are_done:
                values = self.ip()(self._scale(points))
            else:
                # Without the bounds the interpolation cannot be done properly,
                # so we just set everything to zero.
                values = np.zeros((len(points), self.vdim))
            return points, values
        return np.zeros((0, 2)), np.zeros((0, self.vdim), dtype=float)

    def _data_combined(self):
        points, values = self._data_in_bounds()
        if not self.pending_points:
            return points, values
        points_interp, values_interp = self._data_interp()
        points_combined = np.vstack([points, points_interp])
        values_combined = np.vstack([values, values_interp])
        return points_combined, values_combined

    def data_combined(self):
        """Like `data`, however this includes the points in
        `pending_points` for which the values are interpolated."""
        # Interpolate the unfinished points
        points, values = self._data_combined()
        return {tuple(k): v for k, v in zip(points, values)}

    def ip(self):
        """A `scipy.interpolate.LinearNDInterpolator` instance
        containing the learner's data."""
        if self._ip is None:
            points, values = self._data_in_bounds()
            points = self._scale(points)
            self._ip = interpolate.LinearNDInterpolator(points, values)
        return self._ip

    def ip_combined(self):
        """A `scipy.interpolate.LinearNDInterpolator` instance
        containing the learner's data *and* interpolated data of
        the `pending_points`."""
        if self._ip_combined is None:
            points, values = self._data_combined()
            points = self._scale(points)
            self._ip_combined = interpolate.LinearNDInterpolator(points,
                                                                 values)
        return self._ip_combined

    def inside_bounds(self, xy):
        x, y = xy
        (xmin, xmax), (ymin, ymax) = self.bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def tell(self, point, value):
        point = tuple(point)
        self.data[point] = value
        if not self.inside_bounds(point):
            return
        self.pending_points.discard(point)
        self._ip = None
        self._stack.pop(point, None)

    def tell_pending(self, point):
        point = tuple(point)
        if not self.inside_bounds(point):
            return
        self.pending_points.add(point)
        self._ip_combined = None
        self._stack.pop(point, None)

    def _fill_stack(self, stack_till=1):
        if len(self.data) + len(self.pending_points) < self.ndim + 1:
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
            point_new = tuple(self._unscale(point_new))
            loss_new = losses[jsimplex]

            points_new.append(point_new)
            losses_new.append(loss_new)

            self._stack[point_new] = loss_new

            if len(self._stack) >= stack_till:
                break
            else:
                losses[jsimplex] = -np.inf

        return points_new, losses_new

    def ask(self, n, tell_pending=True):
        # Even if tell_pending is False we add the point such that _fill_stack
        # will return new points, later we remove these points if needed.
        points = list(self._stack.keys())
        loss_improvements = list(self._stack.values())
        n_left = n - len(points)
        for p in points[:n]:
            self.tell_pending(p)

        while n_left > 0:
            # The while loop is needed because `stack_till` could be larger
            # than the number of triangles between the points. Therefore
            # it could fill up till a length smaller than `stack_till`.
            new_points, new_loss_improvements = self._fill_stack(
                stack_till=max(n_left, self.stack_size))
            for p in new_points[:n_left]:
                self.tell_pending(p)
            n_left -= len(new_points)

            points += new_points
            loss_improvements += new_loss_improvements

        if not tell_pending:
            self._stack = OrderedDict(zip(points[:self.stack_size],
                                          loss_improvements))
            for point in points[:n]:
                self.pending_points.discard(point)

        return points[:n], loss_improvements[:n]

    @cache_latest
    def loss(self, real=True):
        if not self.bounds_are_done:
            return np.inf
        ip = self.ip() if real else self.ip_combined()
        losses = self.loss_per_triangle(ip)
        return losses.max()

    def remove_unfinished(self):
        self.pending_points = set()
        for p in self._bounds_points:
            if p not in self.data:
                self._stack[p] = np.inf

    def plot(self, n=None, tri_alpha=0):
        """Plot the Learner2D's current state.

        This plot function interpolates the data on a regular grid.
        The gridspacing is evaluated by checking the size of the smallest
        triangle.

        Parameters
        ----------
        n : int
            Number of points in x and y. If None (default) this number is
            evaluated by looking at the size of the smallest triangle.
        tri_alpha : float
            The opacity ``(0 <= tri_alpha <= 1)`` of the triangles overlayed
            on top of the image. By default the triangulation is not visible.

        Returns
        -------
        plot : `holoviews.core.Overlay` or `holoviews.core.HoloMap`
            A `holoviews.core.Overlay` of
            ``holoviews.Image * holoviews.EdgePaths``. If the
            `learner.function` returns a vector output, a
            `holoviews.core.HoloMap` of the
            `holoviews.core.Overlay`\s wil be returned.

        Notes
        -----
        The plot object that is returned if ``learner.function`` returns a
        vector *cannot* be used with the live_plotting functionality.
        """
        hv = ensure_holoviews()
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
            z = ip(x[:, None], y[None, :] * self.aspect_ratio).squeeze()

            if self.vdim > 1:
                ims = {i: hv.Image(np.rot90(z[:, :, i]), bounds=lbrt)
                       for i in range(z.shape[-1])}
                im = hv.HoloMap(ims)
            else:
                im = hv.Image(np.rot90(z), bounds=lbrt)

            if tri_alpha:
                points = self._unscale(ip.tri.points[ip.tri.vertices])
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

    def _get_data(self):
        return self.data

    def _set_data(self, data):
        self.data = data
        # Remove points from stack if they already exist
        for point in copy(self._stack):
            if point in self.data:
                self._stack.pop(point)
