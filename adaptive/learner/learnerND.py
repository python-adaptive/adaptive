# -*- coding: utf-8 -*-
from collections import OrderedDict
import itertools
from math import sqrt

import numpy as np
from scipy import interpolate
import scipy.spatial

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner


def volume(simplex, ys=None):
    matrix = simplex[:-1, :] - simplex[-1, None, :]
    dim = len(simplex) - 1

    # See https://www.jstor.org/stable/2315353
    vol = np.abs(np.linalg.det(matrix)) / np.math.factorial(dim)
    return vol

def orientation(simplex):
    matrix = simplex[:-1, :] - simplex[-1, None, :]
    # See https://www.jstor.org/stable/2315353
    sign, logdet = np.linalg.slogdet(matrix)
    return sign

# translated from https://github.com/mikolalysenko/robust-point-in-simplex/blob/master/rpis.js
# also used https://stackoverflow.com/questions/21819132/how-do-i-check-if-a-simplex-contains-the-origin
def point_in_simplex(simplex, point):
    simplex_orientation = orientation(simplex)
    simplex_copy = simplex[:]
    boundary = False

    for i in range(len(simplex)):
        old_value = simplex[i]
        simplex_copy[i] = point

        orient = orientation(simplex)
        if orient == 0:
            boundary = True
        elif orient != simplex_orientation:
            return -1
        simplex_copy[i] = old_value

    if boundary:
        return 0
    return 1


def uniform_loss(simplex, ys=None):
    return volumes(simplex)

def std_loss(simplex, ys):
    r = np.std(ys, axis=0)
    vol = volume(simplex)

    dim = len(simplex) - 1

    return r.flat * np.power(vol, 1./dim) + vol

def default_loss(simplex, ys):
    return std_loss(simplex, ys)

# # Learner2D and helper functions.
# def volumes(ip):
#     p = ip.tri.points[ip.tri.vertices]
#     matrices = p[:, :-1, :] - p[:, -1, None, :]
#     n_points, dim = ip.tri.points.shape

#     # See https://www.jstor.org/stable/2315353
#     vols = np.abs(np.linalg.det(matrices)) / np.math.factorial(dim)

#     return vols


# def uniform_loss(ip):
#     """Loss function that samples the domain uniformly.

#     Works with `~adaptive.LearnerND` only.

#     Examples
#     --------
#     >>> def f(xy):
#     ...     x, y = xy
#     ...     return x**2 + y**2
#     >>>
#     >>> learner = adaptive.LearnerND(f,
#     ...                              bounds=[(-1, -1), (1, 1)],
#     ...                              loss_per_simplex=uniform_loss)
#     >>>
#     """
#     return volumes(ip)


# def std_loss(ip):
#     # p = ip.tri.points[ip.tri.vertices]
#     # matrices = p[:, :-1, :] - p[:, -1, None, :]
#     v = ip.values[ip.tri.vertices]
#     r = np.std(v, axis=1)
#     vol = volumes(ip)

#     n_points, dim = ip.tri.points.shape

#     return r.flat * np.power(vol, 1./dim) + vol


# def default_loss(ip):
#     return std_loss(ip)


def choose_point_in_simplex(simplex):
    """Choose a new point in inside a simplex.

    Pick the center of the longest edge of this simplex

    Parameters
    ----------
    simplex : numpy array
        The coordinates of a triangle with shape (N+1, N)

    Returns
    -------
    point : numpy array
        The coordinates of the suggested new point.
    """

    # TODO find a better selection algorithm
    longest = 0
    point = None
    N = simplex.shape[1]

    for i in range(2, N+1):
        for j in range(i):
            length = np.linalg.norm(simplex[i, :] - simplex[j, :])
            if length > longest:
                longest = length
                point = (simplex[i, :] + simplex[j, :]) / 2

    return point


class LearnerND(BaseLearner):
    """Learns and predicts a function 'f: ℝ^N → ℝ^M'.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a tuple of two real
        parameters and return a real number.
    bounds : list of 2-tuples
        A list ``[(a1, b1), (a2, b2), ...]`` containing bounds,
        one per dimension.
    loss_per_simplex : callable, optional
        A function that returns the loss for every triangle.
        If not provided, then a default is used, which uses
        the deviation from a linear estimate, as well as
        triangle area, to determine the loss. See the notes
        for more details.


    Attributes
    ----------
    data : dict
        Sampled points and values.

    Methods
    -------
    plotSlice(x_1, x_2, ..., x_n)
        plot a slice of the function using the current data.

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
    These can be used to compute the loss.
    """

    def __init__(self, function, bounds, loss_per_simplex=None):
        self.ndim = len(bounds)
        self._vdim = None
        self.loss_per_simplex = loss_per_simplex or default_loss
        self.bounds = tuple(tuple(map(float, b)) for b in bounds)
        self.data = OrderedDict()
        self._pending: set = set()

        self._mean: float = np.mean(self.bounds, axis=1)
        self._ptp_scale: float = np.ptp(self.bounds, axis=1)

        self._bounds_points = list(itertools.product(*bounds))

        self.function = function
        self._ip = None
        self._loss = np.inf
        self._losses = dict()


    def _scale(self, points):
        # this function converts the points from real coordinates to equalised coordinates,
        # in order to make the triangulation fair
        points = np.asarray(points, dtype=float)
        return (points - self._mean) / self._ptp_scale

    def _unscale(self, points):
        # this functions converts the points from equalised coordinates to real coordinates
        points = np.asarray(points, dtype=float)
        return points * self._ptp_scale + self._mean

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
        return all(p in self.data for p in self._bounds_points)

    def ip(self):
        # raise DeprecationWarning('usage of LinearNDInterpolator should be reduced')
        # returns a scipy.interpolate.LinearNDInterpolator object with the given data as sources
        if self._ip is None:
            points = self._scale(list(self.data.keys()))
            values = np.array(list(self.data.values()), dtype=float)
            self._ip = interpolate.LinearNDInterpolator(points, values)
        return self._ip

    def _tell(self, point, value):
        point = tuple(point)

        if value is None:
            self._pending.add(point)
        else:
            self.data[point] = value
            self._pending.discard(point)
            self._ip = None

    def ask(self, n=1, tell=True):
        # Complexity: O(N log N + n * N)
        # TODO adapt this function
        # TODO allow cases where n > 1

        new_points = []
        new_loss_improvements = []
        if not self.bounds_are_done:
            bounds_to_do = [p for p in self._bounds_points if p not in self.data and p not in self._pending]
            new_points = bounds_to_do[:n]
            new_loss_improvements = [-np.inf] * n
            n = n - len(new_points)

        if n > 0:
            # Interpolate
            ip = self.ip()  # O(N log N) for triangulation
            losses = list(self.losses_combined().items())  # O(N), compute the losses of all interpolated triangles
            losses.sort(key=lambda item: -item[1])  # O(N log N), sort by loss
            for i in range(n):
                simplex, loss = losses[i]  # O(1), Find the index of the simplex with the highest loss
                # simplex = ip.tri.points[ip.tri.vertices[simplex_index]]  # get the corner points the the worst simplex
                point_new = choose_point_in_simplex(np.array(simplex))  # choose a new point in the triangle
                point_new = tuple(self._unscale(point_new))  # relative coordinates to real coordinates
                # loss_new = losses[simplex_index]

                new_points.append(point_new)
                new_loss_improvements.append(loss)

                losses[i] = -np.inf

        if tell:
            self.tell(new_points, itertools.repeat(None))

        return new_points, new_loss_improvements


    # return a dict of simplex -> loss
    def losses(self):
        ip = self.ip()  # TODO: why do we even need the interpolator, just the triangulation would be sufficient
        ret = dict()
        for vertices in ip.tri.vertices: # O(N)
            simplex = ip.tri.points[vertices]
            values = ip.values[vertices]
            key = self._simplex_to_key(simplex)
            if key in self._losses:
                loss = self._losses[key]
            else:
                loss = self.loss_per_simplex(simplex, values)
            ret[key] = loss

        self._losses = ret
        return self._losses

    # TODO make this a global method
    def _simplex_to_key(self, simplex):
        l = simplex.tolist()
        l.sort()
        l = tuple(map(tuple, simplex))
        return l

    # return a dict of simplex -> loss
    def losses_combined(self):
        if self.bounds_are_done == False:
            return self.losses()  # TODO find a better metric, like, triangulate everything and give it infinite loss
        # TODO also fix that we can request new points even if there does not even if all points are pending
        # assume the number of pending points is reasonably low (eg 20 or so) to keep performance high
        # TODO actually make performance better
        ip = self.ip()  # O(N log N)
        losses = self.losses()  # O(N log N), this also has a slow python loop
        # now start adding new items to it, yay
        pending = list(self._pending)  # O(P) let P be the number of pending points

        pending_points_per_simplex = dict()  # TODO better name

        # TODO come up with better name
        simplices = ip.tri.find_simplex(pending)  # O(N*P) I guess, find the simplex each point belongs to
        # TODO what happens if a point belongs to multiple simplices 
        # (like when it lies on an edge, we should make sure that gets handled as well)

        # TODO take a look at ip.tri.neighbors[i] as this might be interesting

        for i in range(len(simplices)):
            s = self._simplex_to_key(ip.tri.points[ip.tri.vertices[simplices[i]]])
            pending_points_per_simplex[s] = pending_points_per_simplex.get(s, [])
            pending_points_per_simplex[s].append(pending[i])

        for simplex, pending in pending_points_per_simplex.items():
            # Get all vertices in this simplex (including the known border)
            vertices = np.append(simplex, pending, axis=0)
            # Triangulate the vertices
            print(vertices)
            triangulation = scipy.spatial.Delaunay(vertices)
            pending_simplices = vertices[triangulation.simplices]

            total_volume = volume(np.array(simplex))
            loss_per_volume = losses[simplex] / total_volume
            losses[simplex] = 0  # do not use this simplex, only it's children
            for simp in pending_simplices:
                key = self._simplex_to_key(simp)
                vol = volume(simp)
                losses[key] = loss_per_volume * vol

        return losses  # TODO actually get losses_combined

    def loss(self, real=True):
        losses = self.losses() if real else self.losses_combined()
        if len(losses) == 0:
            return float('inf')
        else:
            return max(losses.values())

    def remove_unfinished(self):
        self._pending = set()

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
                # n = int(0.658 / sqrt(volumes(ip).min()))
                n = 50

            x = y = np.linspace(-0.5, 0.5, n)
            z = ip(x[:, None], y[None, :]).squeeze()

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

    def plot_slice(self, values, n=None):
        values = list(values)
        count_none = values.count(None)
        assert(count_none == 1 or count_none == 2)
        if count_none == 2:
            hv = ensure_holoviews()
            if self.vdim > 1:
                raise NotImplemented('holoviews currently does not support',
                                     '3D surface plots in bokeh.')

            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                # n = int(0.658 / sqrt(volumes(ip).min()))  # TODO fix this calculation
                n = 50

            x = y = np.linspace(-0.5, 0.5, n)
            x = x[:, None]
            y = y[None, :]
            i = values.index(None)
            values[i] = 0
            j = values.index(None)
            values[j] = y
            values[i] = x
            bx, by = self.bounds[i], self.bounds[j]
            lbrt = bx[0], by[0], bx[1], by[1]

            if len(self.data) >= 4:
                ip = self.ip()
                z = ip(*values).squeeze()

                im = hv.Image(np.rot90(z), bounds=lbrt)
            else:
                im = hv.Image([], bounds=lbrt)

            im_opts = dict(cmap='viridis')

            return im.opts(style=im_opts)
        else:
            hv = ensure_holoviews()
            if not self.data:
                p = hv.Scatter([]) * hv.Path([])
            elif not self.vdim > 1:
                ind = values.index(None)
                if n is None:
                    n = 500
                x = np.linspace(-0.5, 0.5, n)
                values[ind] = 0
                values = list(self._scale(values))
                values[ind] = x
                ip = self.ip()
                y = ip(*values)
                x = x * self._ptp_scale[ind] + self._mean[ind]
                p = hv.Path((x, y))
            else:
                raise NotImplementedError('multidimensional output not yet supported by plotSlice')

            # Plot with 5% empty margins such that the boundary points are visible
            margin = 0.05 * self._ptp_scale[ind]
            plot_bounds = (x[0] - margin, x[-1] + margin)

            return p.redim(x=dict(range=plot_bounds))
