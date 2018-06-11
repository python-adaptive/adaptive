# -*- coding: utf-8 -*-
from collections import OrderedDict
import itertools
import heapq

import numpy as np
from scipy import interpolate
import scipy.spatial

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner
import copy

from .triangulation import Triangulation


def find_initial_simplex(pts, ndim):
    origin = pts[0]
    vecs = pts[1:] - origin
    if np.linalg.matrix_rank(vecs) < ndim:
        return None  # all points are coplanar

    tri = scipy.spatial.Delaunay(pts)
    simplex = tri.simplices[0] # take a random simplex from tri
    return simplex


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
    simplex_copy = copy.deepcopy(simplex)
    simplex_orientation = orientation(simplex_copy)
    boundary = False

    for i in range(len(simplex)):
        simplex_copy[i] = point

        orient = orientation(simplex_copy)
        if orient == 0:
            boundary = True
        elif orient != simplex_orientation:
            return -1
        simplex_copy[i] = copy.copy(simplex[i])

    if boundary:
        return 0
    return 1


def uniform_loss(simplex, ys=None):
    return volume(simplex)


def std_loss(simplex, ys):
    r = np.std(ys, axis=0)
    vol = volume(simplex)

    dim = len(simplex) - 1

    return r.flat * np.power(vol, 1./dim) + vol


def default_loss(simplex, ys):
    return std_loss(simplex, ys)


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
        self._tri = None
        self._losses = dict()

        self._vertex_to_simplex_cache = dict()

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
        # TODO take our own triangulation into account when generating the ip
        points = self._scale(list(self.data.keys()))
        values = np.array(list(self.data.values()), dtype=float)
        return interpolate.LinearNDInterpolator(points, values)

    def tri(self):
        if self._tri is None:
            if len(self.data) < 2:
                return None
            points = self._scale(list(self.data.keys()))
            initial_simplex = find_initial_simplex(points, self.ndim)
            if initial_simplex is None:
                return None
            pts = points[initial_simplex]
            self._tri = Triangulation(pts)
            to_add = [points[i] for i in range(len(points)) if i not in initial_simplex]

            for p in to_add:
                self._tri.add_point(p)

        return self._tri

    def values(self):
        return np.array(list(self.data.values()), dtype=float)

    _time = None
    _prev = 1/30

    def time(self):
        import time
        if self._time is None:
            self._time = time.time()
        self._prev = 0.98 * self._prev + (time.time() - self._time) * 0.02
        self._time = time.time()
        return self._prev

    def _tell(self, point, value):
        point = tuple(point)

        if value is None:
            self._pending.add(point)
        else:
            self._pending.discard(point)

            print("addpoint", self.npoints, ":", "(p/s: %.2f)" % (1/self.time()), point)
            self.data[point] = value

            if self._tri is not None:
                simplex = self._vertex_to_simplex_cache.get(point, None)
                if not self._simplex_exists(simplex):
                    simplex = None
                self._tri.add_point(self._scale(point), simplex)

    def _simplex_exists(self, simplex):
        tri = self.tri()
        simplex = tuple(sorted(simplex))
        return simplex in tri.simplices

    def volume(self):
        tri = self.tri()
        if tri is None:
            return 0
        v = [volume(self._unscale([tri.vertices[i] for i in s])) for s in tri.simplices]
        v = sum(v)
        return v

    def ask(self, n=1, tell=True):
        assert tell
        xs = []
        losses = []
        for i in range(n):
            x, loss = self._ask()
            xs.append(*x)
            losses.append(*loss)
        return xs, losses

    def _ask(self, n=1, tell=True):
        # Complexity: O(N log N + n * N)
        # TODO adapt this function
        # TODO allow cases where n > 1

        new_points = []
        new_loss_improvements = []
        if not self.bounds_are_done:
            bounds_to_do = [p for p in self._bounds_points if p not in self.data and p not in self._pending]
            new_points = bounds_to_do[:n]
            new_loss_improvements = [-np.inf] * len(new_points)
            n = n - len(new_points)

        if n > 0:
            losses = [(-v, k) for k,v in self.losses().items()]
            tri = self.tri()
            heapq.heapify(losses)

            while len(new_points) < n:
                loss, simplex = heapq.heappop(losses)
                # verify that simplex still exists
                if not self._simplex_exists(simplex):
                    # it could be that some of the points are pending, then always accept
                    npoints = len(self.data)
                    if all([i < npoints for i in simplex]):
                        self._losses.pop(simplex, None)
                        continue

                points = np.array([tri.vertices[i] for i in simplex])
                point_new = choose_point_in_simplex(points)  # choose a new point in the simplex
                point_new = tuple(self._unscale(point_new))  # relative coordinates to real coordinates
                self._vertex_to_simplex_cache[point_new] = simplex

                new_points.append(point_new)
                new_loss_improvements.append(loss)

        if tell:
            self.tell(new_points, itertools.repeat(None))

        return new_points, new_loss_improvements

    def losses(self):
        """
        :return: a dict of simplex -> loss
        """
        if self.tri() is None:
            return dict()

        tri = self.tri()
        if tri is not None:
            losses_to_compute = tri.simplices - set(self._losses)
            for simplex in losses_to_compute:
                vertices = self._unscale([tri.vertices[i] for i in simplex])
                values = [self.data[tuple(self._unscale(tri.vertices[i]))] for i in simplex]
                loss = self.loss_per_simplex(vertices, values)
                self._losses[simplex] = float(loss)


            losses_to_delete = set(self._losses) - tri.simplices
            for simplex in losses_to_delete:
                del self._losses[simplex]

        return self._losses

    def losses_combined(self):
        """
        :return: a dict of simplex -> loss
        """
        raise NotImplementedError("wait some while")
        if self.bounds_are_done == False:
            ret = dict()
            pts = self._scale(list(self.data.keys()) + list(self._pending))
            tri = scipy.spatial.Delaunay(pts)
            for vertices in tri.vertices:
                simplex = tri.points[vertices]
                key = self._simplex_to_key(simplex)
                ret[key] = np.inf
            return ret

        if len(self._pending) == 0:
            return self.losses()
        # assume the number of pending points is reasonably low (eg 20 or so) to keep performance high
        # TODO actually make performance better
        tri = self.tri()  # O(1)
        losses = self._losses.copy()
        # now start adding new items to it, yay
        pending = self._scale(list(self._pending))  # O(P) let P be the number of pending points

        pending_points_per_simplex = dict()  # TODO better name

        # TODO come up with better name
        simplices = tri.find_simplex(pending)  # O(N*P) I guess, find the simplex each point belongs to
        # TODO what happens if a point belongs to multiple simplices 
        # (like when it lies on an edge, we should make sure that gets handled as well)

        for index in range(len(simplices)):
            pending_point = pending[index]

            # do a DFS of all neighbouring simplices to find which simplices contain the point
            stack = [simplices[index]]  # contains all done and pending simplices
            ind = 0
            while ind < len(stack):
                q = stack[ind]
                s = tri.points[tri.vertices[q]]
                if point_in_simplex(s, pending_point) >= 0:
                    k = self._simplex_to_key(s)
                    pending_points_per_simplex[k] = pending_points_per_simplex.get(k, [])
                    pending_points_per_simplex[k].append(pending_point)
                    for neighbor in tri.neighbors[q]:
                        if neighbor >= 0 and neighbor not in stack:
                            stack.append(neighbor)
                ind += 1


        for simplex, pending in pending_points_per_simplex.items():
            # Get all vertices in this simplex (including the known border)
            vertices = np.append(simplex, pending, axis=0)
            # Triangulate the vertices
            if volume(np.asarray(simplex)) < 1e-16:
                continue

            triangulation = scipy.spatial.Delaunay(vertices)
            pending_simplices = vertices[triangulation.simplices]

            total_volume = volume(np.array(simplex))
            if simplex not in losses:
                values = [self.data.get(tuple(x)) for x in self._unscale(simplex)]
                loss = self.loss_per_simplex(np.array(simplex), values)
                losses[simplex] = loss
                self._losses[simplex] = loss
            loss_per_volume = losses[simplex] / total_volume
            # <class 'tuple'>: ((0.0, -0.25), (0.125, -0.25), (0.1875, -0.125))
            losses.pop(simplex)  # do not use this simplex, only it's children
            for simp in pending_simplices:
                key = self._simplex_to_key(simp)
                vol = volume(simp)
                losses[key] = loss_per_volume * vol

        return losses

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
                tri = self.tri()
                points = np.array([[tri.vertices[i] for i in s] for s in tri.simplices])
                points = self._unscale(points)
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
