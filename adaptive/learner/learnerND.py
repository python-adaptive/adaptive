# -*- coding: utf-8 -*-
from collections import OrderedDict, Iterable
import heapq
import itertools
import random

import numpy as np
from scipy import interpolate
import scipy.spatial

from ..notebook_integration import ensure_holoviews
from .base_learner import BaseLearner

from .triangulation import Triangulation, point_in_simplex, \
                           circumsphere, simplex_volume_in_embedding


def volume(simplex, ys=None):
    # Notice the parameter ys is there so you can use this volume method as
    # as loss function
    matrix = np.array(np.subtract(simplex[:-1], simplex[-1]), dtype=float)
    dim = len(simplex) - 1

    # See https://www.jstor.org/stable/2315353
    vol = np.abs(np.linalg.det(matrix)) / np.math.factorial(dim)
    return vol


def orientation(simplex):
    matrix = np.subtract(simplex[:-1], simplex[-1])
    # See https://www.jstor.org/stable/2315353
    sign, _logdet = np.linalg.slogdet(matrix)
    return sign


def uniform_loss(simplex, ys=None):
    return volume(simplex)


def std_loss(simplex, ys):
    r = np.linalg.norm(np.std(ys, axis=0))
    vol = volume(simplex)

    dim = len(simplex) - 1

    return r.flat * np.power(vol, 1. / dim) + vol


def default_loss(simplex, ys):
    # return std_loss(simplex, ys)
    if isinstance(ys[0], Iterable):
        pts = [(*x, *y) for x, y in zip(simplex, ys)]
    else:
        pts = [(*x, y) for x, y in zip(simplex, ys)]
    return simplex_volume_in_embedding(pts)


def choose_point_in_simplex(simplex, transform=None):
    """Choose a new point in inside a simplex.

    Pick the center of the simplex if the shape is nice (that is, the 
    circumcenter lies within the simplex). Otherwise take the middle of the 
    longest edge.

    Parameters
    ----------
    simplex : numpy array
        The coordinates of a triangle with shape (N+1, N)
    transform : N*N matrix
        The multiplication to apply to the simplex before choosing the new point

    Returns
    -------
    point : numpy array of length N
        The coordinates of the suggested new point.
    """

    if transform is not None:
        simplex = np.dot(simplex, transform)

    # choose center if and only if the shape of the simplex is nice,
    # otherwise: the center the longest edge
    center, _radius = circumsphere(simplex)
    if point_in_simplex(center, simplex):
        point = np.average(simplex, axis=0)
    else:
        distances = scipy.spatial.distance.pdist(simplex)
        distance_matrix = scipy.spatial.distance.squareform(distances)
        i, j = np.unravel_index(np.argmax(distance_matrix),
                                distance_matrix.shape)

        point = (simplex[i, :] + simplex[j, :]) / 2

    if transform is not None:
        point = np.linalg.solve(transform, point)  # undo the transform
        
    return point


class LearnerND(BaseLearner):
    """Learns and predicts a function 'f: ℝ^N → ℝ^M'.

    Parameters
    ----------
    func: callable
        The function to learn. Must take a tuple of N real
        parameters and return a real number or an arraylike of length M.
    bounds : list of 2-tuples
        A list ``[(a_1, b_1), (a_2, b_2), ..., (a_n, b_n)]`` containing bounds,
        one pair per dimension.
    loss_per_simplex : callable, optional
        A function that returns the loss for a simplex.
        If not provided, then a default is used, which uses
        the deviation from a linear estimate, as well as
        triangle area, to determine the loss.


    Attributes
    ----------
    data : dict
        Sampled points and values.
    points : numpy array
        Coordinates of the currently known points
    values : numpy array
        The values of each of the known points

    Methods
    -------
    plot(n)
        If dim == 2, this method will plot the function being learned.
    plot_slice(cut_mapping, n)
        plot a slice of the function using interpolation of the current data.
        the cut_mapping contains the fixed parameters, the other parameters are
        used as axes for plotting.

    Notes
    -----
    The sample points are chosen by estimating the point where the
    gradient is maximal. This is based on the currently known points.

    In practice, this sampling protocol results to sparser sampling of
    flat regions, and denser sampling of regions where the function
    has a high gradient, which is useful if the function is expensive to
    compute.

    This sampling procedure is not fast, so to benefit from
    it, your function needs to be slow enough to compute.


    This class keeps track of all known points. It triangulates these points
    and with every simplex it associates a loss. Then if you request points
    that you will compute in the future, it will subtriangulate a real simplex
    with the pending points inside it and distribute the loss among it's
    children based on volume.
    """

    def __init__(self, func, bounds, loss_per_simplex=None):
        self.ndim = len(bounds)
        self._vdim = None
        self.loss_per_simplex = loss_per_simplex or default_loss
        self.bounds = tuple(tuple(map(float, b)) for b in bounds)
        self.data = OrderedDict()
        self._pending = set()

        self._bounds_points = list(map(tuple, itertools.product(*bounds)))

        self.function = func
        self._tri = None
        self._losses = dict()

        self._pending_to_simplex = dict()  # vertex -> simplex

        # triangulation of the pending points inside a specific simplex
        self._subtriangulations = dict()  # simplex -> triangulation

        # scale to unit
        self._transform = np.linalg.inv(np.diag(np.diff(bounds).flat))

        # create a private random number generator with fixed seed
        self._random = random.Random(1)

        # all real triangles that have not been subdivided and the pending 
        # triangles heap of tuples (-loss, real simplex, sub_simplex or None)

        # _simplex_queue is a heap of tuples (-loss, real_simplex, sub_simplex)
        # It contains all real and pending simplices except for real simplices
        # that have been subdivided.
        # _simplex_queue may contain simplices that have been deleted, this is
        #  because deleting those items from the heap is an expensive operation,
        # so when popping an item, you should check that the simplex that has
        # been returned has not been deleted. This checking is done by
        # _pop_highest_existing_simplex
        self._simplex_queue = []  # heap

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
        # XXX: take our own triangulation into account when generating the ip
        return interpolate.LinearNDInterpolator(self.points, self.values)

    @property
    def tri(self):
        if self._tri is not None:
            return self._tri

        try:
            self._tri = Triangulation(self.points)
            return self._tri
        except ValueError:
            # A ValueError is raised if we do not have enough points or
            # the provided points are coplanar, so we need more points to create
            # a valid triangulation
            return None

        # XXX: also compute losses of initial simplex

    @property
    def values(self):
        return np.array(list(self.data.values()), dtype=float)

    @property
    def points(self):
        return np.array(list(self.data.keys()), dtype=float)

    def tell(self, point, value):
        point = tuple(point)

        if point in self.data:
            return  # we already know about the point

        if value is None:
            return self._tell_pending(point)

        self._pending.discard(point)
        tri = self.tri
        self.data[point] = value

        if tri is not None:
            simplex = self._pending_to_simplex.get(point)
            if simplex is not None and not self._simplex_exists(simplex):
                simplex = None
            to_delete, to_add = tri.add_point(
                point, simplex, transform=self._transform)
            self.update_losses(to_delete, to_add)

    def _simplex_exists(self, simplex):
        simplex = tuple(sorted(simplex))
        return simplex in self.tri.simplices

    def _tell_pending(self, point, simplex=None):
        point = tuple(point)
        self._pending.add(point)

        if self.tri is None:
            return

        simplex = tuple(simplex or self.tri.locate_point(point))
        if not simplex:
            return
            # Simplex is None if pending point is outside the triangulation,
            # then you do not have subtriangles

        simplex = tuple(simplex)
        simplices = [self.tri.vertex_to_simplices[i] for i in simplex]
        neighbours = set.union(*simplices)
        # Neighbours also includes the simplex itself

        for simpl in neighbours:
            _, to_add = self._try_adding_pending_point_to_simplex(point, simpl)
            if to_add is None:
                continue
            self._update_subsimplex_losses(simpl, to_add)

    def _try_adding_pending_point_to_simplex(self, point, simplex):
        # try to insert it
        if not self.tri.point_in_simplex(point, simplex):
            return None, None

        if simplex not in self._subtriangulations:
            vertices = self.tri.get_vertices(simplex)
            self._subtriangulations[simplex] = Triangulation(vertices)

        self._pending_to_simplex[point] = simplex
        return self._subtriangulations[simplex].add_point(point)

    def _update_subsimplex_losses(self, simplex, new_subsimplices):
        loss = self._losses[simplex]

        loss_density = loss / self.tri.volume(simplex)
        subtriangulation = self._subtriangulations[simplex]
        for subsimplex in new_subsimplices:
            subloss = subtriangulation.volume(subsimplex) * loss_density
            heapq.heappush(self._simplex_queue,
                           (-subloss, simplex, subsimplex))

    def ask(self, n=1):
        xs, losses = zip(*(self._ask() for _ in range(n)))
        return list(xs), list(losses)

    def _ask_bound_point(self):
        # get the next bound point that is still available
        new_point = next(p for p in self._bounds_points
                         if p not in self.data and p not in self._pending)
        self._tell_pending(new_point)
        return new_point, np.inf

    def _ask_point_without_known_simplices(self):
        assert not self._bounds_available
        # pick a random point inside the bounds
        # XXX: change this into picking a point based on volume loss
        a = np.diff(self.bounds).flat
        b = np.array(self.bounds)[:, 0]
        r = np.array([self._random.random() for _ in range(self.ndim)])
        p = r * a + b
        p = tuple(p)

        self._tell_pending(p)
        return p, np.inf

    def _pop_highest_existing_simplex(self):
        # find the simplex with the highest loss, we do need to check that the
        # simplex hasn't been deleted yet
        while len(self._simplex_queue):
            loss, simplex, subsimplex = heapq.heappop(self._simplex_queue) 
            if (subsimplex is None
                    and simplex in self.tri.simplices
                    and simplex not in self._subtriangulations):
                return abs(loss), simplex, subsimplex
            if (simplex in self._subtriangulations
                    and simplex in self.tri.simplices
                    and subsimplex in self._subtriangulations[simplex].simplices):
                return abs(loss), simplex, subsimplex

        # Could not find a simplex, this code should never be reached
        assert self.tri is not None
        raise AssertionError(
            """Could not find a simplex to. Yet there should always be a simplex 
            available if LearnerND.tri() is not None"""
        )

    def _ask_best_point(self):
        assert self.tri is not None

        loss, simplex, subsimplex = self._pop_highest_existing_simplex()

        if subsimplex is None:
            # We found a real simplex and want to subdivide it
            points = self.tri.get_vertices(simplex)
        else:
            # We found a pending simplex and want to subdivide it
            subtri = self._subtriangulations[simplex]
            points = subtri.get_vertices(subsimplex)

        point_new = tuple(choose_point_in_simplex(points,
                                                  transform=self._transform))

        self._pending_to_simplex[point_new] = simplex
        self._tell_pending(point_new, simplex)  # O(??)

        return point_new, loss

    @property
    def _bounds_available(self):
        return any((p not in self._pending and p not in self.data)
                   for p in self._bounds_points)

    def _ask(self):
        if self._bounds_available:
            return self._ask_bound_point()  # O(1)

        if self.tri is None:
            # All bound points are pending or have been evaluated, but we do not
            # have enough evaluated points to construct a triangulation, so we
            # pick a random point
            return self._ask_point_without_known_simplices()  # O(1)

        return self._ask_best_point()  # O(log N)

    def update_losses(self, to_delete: set, to_add: set):
        # XXX: add the points outside the triangulation to this as well
        pending_points_unbound = set()

        for simplex in to_delete:
            loss = self._losses.pop(simplex, None)
            subtri = self._subtriangulations.pop(simplex, None)
            if subtri is not None:
                pending_points_unbound.update(subtri.vertices)

        pending_points_unbound = set(p for p in pending_points_unbound
                                     if p not in self.data)

        for simplex in to_add:
            vertices = self.tri.get_vertices(simplex)
            values = [self.data[tuple(v)] for v in vertices]
            loss = float(self.loss_per_simplex(vertices, values))
            self._losses[simplex] = float(loss)

            for p in pending_points_unbound:
                self._try_adding_pending_point_to_simplex(p, simplex)

            if simplex not in self._subtriangulations:
                heapq.heappush(self._simplex_queue, (-loss, simplex, None))
                continue

            self._update_subsimplex_losses(simplex,
                                self._subtriangulations[simplex].simplices)

    def losses(self):
        """Get the losses of each simplex in the current triangulation, as dict

        Returns
        -------
        losses : dict
            the key is a simplex, the value is the loss of this simplex
        """
        # XXX could be a property
        if self.tri is None:
            return dict()

        return self._losses

    def loss(self, real=True):
        losses = self.losses()  # XXX: compute pending loss if real == False
        return max(losses.values()) if losses else float('inf')

    def remove_unfinished(self):
        # XXX: implement this method
        self._pending = set()
        self._subtriangulations = dict()
        self._pending_to_simplex = dict()

    def plot(self, n=None, tri_alpha=0):
        """Plot the function we want to learn, only works in 2D.

        Parameters
        ----------
        n : int
            the number of boxes in the interpolation grid along each axis
        tri_alpha : float (0 to 1)
            Opacity of triangulation lines
        """
        hv = ensure_holoviews()
        if self.vdim > 1:
            raise NotImplementedError('holoviews currently does not support',
                                      '3D surface plots in bokeh.')
        if len(self.bounds) != 2:
            raise NotImplementedError("Only 2D plots are implemented: You can "
                                      "plot a 2D slice with 'plot_slice'.")
        x, y = self.bounds
        lbrt = x[0], y[0], x[1], y[1]

        if len(self.data) >= 4:
            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                n = int(0.658 / np.sqrt(np.min(self.tri.volumes())))

            xs = ys = np.linspace(0, 1, n)
            xs = xs * (x[1] - x[0]) + x[0]
            ys = ys * (y[1] - y[0]) + y[0]
            z = self.ip()(xs[:, None], ys[None, :]).squeeze()

            im = hv.Image(np.rot90(z), bounds=lbrt)

            if tri_alpha:
                points = np.array([self.tri.get_vertices(s)
                                   for s in self.tri.simplices])
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

    def plot_slice(self, cut_mapping, n=None):
        """Plot a 1d or 2d interpolated slice of a N-dimensional function.

        Parameters
        ----------
        cut_mapping : dict (int -> float)
            for each fixed dimension the value, the other dimensions
            are interpolated
        n : int
            the number of boxes in the interpolation grid along each axis
        """
        hv = ensure_holoviews()
        plot_dim = self.ndim - len(cut_mapping)
        if plot_dim == 1:
            if not self.data:
                return hv.Scatter([]) * hv.Path([])
            elif self.vdim > 1:
                raise NotImplementedError('multidimensional output not yet'
                                          ' supported by `plot_slice`')
            n = n or 201
            values = [cut_mapping.get(i, np.linspace(*self.bounds[i], n))
                      for i in range(self.ndim)]
            ind = next(i for i in range(self.ndim) if i not in cut_mapping)
            x = values[ind]
            y = self.ip()(*values)
            p = hv.Path((x, y))

            # Plot with 5% margins such that the boundary points are visible
            margin = 0.05 / self._transform[ind, ind]
            plot_bounds = (x.min() - margin, x.max() + margin)
            return p.redim(x=dict(range=plot_bounds))

        elif plot_dim == 2:
            if self.vdim > 1:
                raise NotImplementedError('holoviews currently does not support'
                                          ' 3D surface plots in bokeh.')
            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                n = int(0.658 / np.sqrt(np.min(self.tri.volumes())))

            xs = ys = np.linspace(0, 1, n)
            xys = [xs[:, None], ys[None, :]]
            values = [cut_mapping[i] if i in cut_mapping
                      else xys.pop(0) * (b[1] - b[0]) + b[0]
                      for i, b in enumerate(self.bounds)]

            lbrt = [b for i, b in enumerate(self.bounds)
                    if i not in cut_mapping]
            lbrt = np.reshape(lbrt, (2, 2)).T.flatten().tolist()

            if len(self.data) >= 4:
                z = self.ip()(*values).squeeze()
                im = hv.Image(np.rot90(z), bounds=lbrt)
            else:
                im = hv.Image([], bounds=lbrt)

            return im.opts(style=dict(cmap='viridis'))
        else:
            raise ValueError("Only 1 or 2-dimensional plots can be generated.")
