from __future__ import annotations

import functools
import itertools
import random
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import scipy.spatial
from scipy import interpolate
from sortedcontainers import SortedKeyList

from adaptive.learner.base_learner import BaseLearner, uses_nth_neighbors
from adaptive.learner.triangulation import (
    Triangulation,
    circumsphere,
    fast_det,
    point_in_simplex,
    simplex_volume_in_embedding,
)
from adaptive.notebook_integration import ensure_holoviews, ensure_plotly
from adaptive.utils import (
    assign_defaults,
    cache_latest,
    partial_function_from_dataframe,
    restore,
)

try:
    import pandas

    with_pandas = True

except ModuleNotFoundError:
    with_pandas = False


def to_list(inp):
    if isinstance(inp, Iterable):
        return list(inp)
    return [inp]


def volume(simplex, ys=None):
    # Notice the parameter ys is there so you can use this volume method as
    # as loss function
    matrix = np.subtract(simplex[:-1], simplex[-1], dtype=float)

    # See https://www.jstor.org/stable/2315353
    dim = len(simplex) - 1
    vol = np.abs(fast_det(matrix)) / np.math.factorial(dim)
    return vol


def orientation(simplex):
    matrix = np.subtract(simplex[:-1], simplex[-1])
    # See https://www.jstor.org/stable/2315353
    sign, _logdet = np.linalg.slogdet(matrix)
    return sign


def uniform_loss(simplex, values, value_scale):
    """
    Uniform loss.

    Parameters
    ----------
    simplex : list of tuples
        Each entry is one point of the simplex.
    values : list of values
        The scaled function values of each of the simplex points.
    value_scale : float
        The scale of values, where ``values = function_values * value_scale``.

    Returns
    -------
    loss : float
    """
    return volume(simplex)


def std_loss(simplex, values, value_scale):
    """
    Computes the loss of the simplex based on the standard deviation.

    Parameters
    ----------
    simplex : list of tuples
        Each entry is one point of the simplex.
    values : list of values
        The scaled function values of each of the simplex points.
    value_scale : float
        The scale of values, where ``values = function_values * value_scale``.

    Returns
    -------
    loss : float
    """

    r = np.linalg.norm(np.std(values, axis=0))
    vol = volume(simplex)

    dim = len(simplex) - 1

    return r.flat * np.power(vol, 1.0 / dim) + vol


def default_loss(simplex, values, value_scale):
    """
    Computes the average of the volumes of the simplex.

    Parameters
    ----------
    simplex : list of tuples
        Each entry is one point of the simplex.
    values : list of values
        The scaled function values of each of the simplex points.
    value_scale : float
        The scale of values, where ``values = function_values * value_scale``.

    Returns
    -------
    loss : float
    """
    if isinstance(values[0], Iterable):
        pts = [(*x, *y) for x, y in zip(simplex, values)]
    else:
        pts = [(*x, y) for x, y in zip(simplex, values)]
    return simplex_volume_in_embedding(pts)


@uses_nth_neighbors(1)
def triangle_loss(simplex, values, value_scale, neighbors, neighbor_values):
    """
    Computes the average of the volumes of the simplex combined with each
    neighbouring point.

    Parameters
    ----------
    simplex : list of tuples
        Each entry is one point of the simplex.
    values : list of values
        The scaled function values of each of the simplex points.
    value_scale : float
        The scale of values, where ``values = function_values * value_scale``.
    neighbors : list of tuples
        The neighboring points of the simplex, ordered such that simplex[0]
        exactly opposes neighbors[0], etc.
    neighbor_values : list of values
        The function values for each of the neighboring points.

    Returns
    -------
    loss : float
    """

    neighbors = [n for n in neighbors if n is not None]
    neighbor_values = [v for v in neighbor_values if v is not None]
    if len(neighbors) == 0:
        return 0

    s = [(*x, *to_list(y)) for x, y in zip(simplex, values)]
    n = [(*x, *to_list(y)) for x, y in zip(neighbors, neighbor_values)]

    return sum(simplex_volume_in_embedding([*s, neighbor]) for neighbor in n) / len(
        neighbors
    )


def curvature_loss_function(exploration=0.05):
    # XXX: add doc-string!
    @uses_nth_neighbors(1)
    def curvature_loss(simplex, values, value_scale, neighbors, neighbor_values):
        """Compute the curvature loss of a simplex.

        Parameters
        ----------
        simplex : list of tuples
            Each entry is one point of the simplex.
        values : list of values
            The scaled function values of each of the simplex points.
        value_scale : float
            The scale of values, where ``values = function_values * value_scale``.
        neighbors : list of tuples
            The neighboring points of the simplex, ordered such that simplex[0]
            exactly opposes neighbors[0], etc.
        neighbor_values : list of values
            The scaled function values for each of the neighboring points.

        Returns
        -------
        loss : float
        """
        dim = len(simplex[0])  # the number of coordinates
        loss_input_volume = volume(simplex)

        loss_curvature = triangle_loss(
            simplex, values, value_scale, neighbors, neighbor_values
        )
        return (
            loss_curvature + exploration * loss_input_volume ** ((2 + dim) / dim)
        ) ** (1 / (2 + dim))

    return curvature_loss


def choose_point_in_simplex(simplex, transform=None):
    """Choose a new point in inside a simplex.

    Pick the center of the simplex if the shape is nice (that is, the
    circumcenter lies within the simplex). Otherwise take the middle of the
    longest edge.

    Parameters
    ----------
    simplex : numpy array
        The coordinates of a triangle with shape (N+1, N).
    transform : N*N matrix
        The multiplication to apply to the simplex before choosing
        the new point.

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
        i, j = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        point = (simplex[i, :] + simplex[j, :]) / 2

    if transform is not None:
        point = np.linalg.solve(transform, point)  # undo the transform

    return point


def _simplex_evaluation_priority(key):
    # We round the loss to 8 digits such that losses
    # are equal up to numerical precision will be considered
    # to be equal. This is needed because we want the learner
    # to behave in a deterministic fashion.
    loss, simplex, subsimplex = key
    return -round(loss, ndigits=8), simplex, subsimplex or (0,)


class LearnerND(BaseLearner):
    """Learns and predicts a function 'f: ℝ^N → ℝ^M'.

    Parameters
    ----------
    func: callable
        The function to learn. Must take a tuple of N real
        parameters and return a real number or an arraylike of length M.
    bounds : list of 2-tuples or `scipy.spatial.ConvexHull`
        A list ``[(a_1, b_1), (a_2, b_2), ..., (a_n, b_n)]`` containing bounds,
        one pair per dimension.
        Or a ConvexHull that defines the boundary of the domain.
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
    pending_points : set
        Points that still have to be evaluated.

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
        self._vdim = None
        self.loss_per_simplex = loss_per_simplex or default_loss

        if hasattr(self.loss_per_simplex, "nth_neighbors"):
            if self.loss_per_simplex.nth_neighbors > 1:
                raise NotImplementedError(
                    "The provided loss function wants "
                    "next-nearest neighboring simplices for the loss computation, "
                    "this feature is not yet implemented, either use "
                    "nth_neightbors = 0 or 1"
                )
            self.nth_neighbors = self.loss_per_simplex.nth_neighbors
        else:
            self.nth_neighbors = 0

        self.data = OrderedDict()
        self.pending_points = set()

        self.bounds = bounds
        if isinstance(bounds, scipy.spatial.ConvexHull):
            hull_points = bounds.points[bounds.vertices]
            self._bounds_points = sorted(list(map(tuple, hull_points)))
            self._bbox = tuple(zip(hull_points.min(axis=0), hull_points.max(axis=0)))
            self._interior = scipy.spatial.Delaunay(self._bounds_points)
        else:
            self._bounds_points = sorted(list(map(tuple, itertools.product(*bounds))))
            self._bbox = tuple(tuple(map(float, b)) for b in bounds)
            self._interior = None

        self.ndim = len(self._bbox)

        self.function = func
        self._tri = None
        self._losses = dict()

        self._pending_to_simplex = dict()  # vertex → simplex

        # triangulation of the pending points inside a specific simplex
        self._subtriangulations = dict()  # simplex → triangulation

        # scale to unit hypercube
        # for the input
        self._transform = np.linalg.inv(np.diag(np.diff(self._bbox).flat))
        # for the output
        self._min_value = None
        self._max_value = None
        self._old_scale = None
        self._output_multiplier = (
            1  # If we do not know anything, do not scale the values
        )
        self._recompute_losses_factor = 1.1

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
        self._simplex_queue = SortedKeyList(key=_simplex_evaluation_priority)

    def new(self) -> LearnerND:
        """Create a new learner with the same function and bounds."""
        return LearnerND(self.function, self.bounds, self.loss_per_simplex)

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
        return self._vdim if self._vdim is not None else 1

    def to_numpy(self):
        """Data as NumPy array of size ``(npoints, dim+vdim)``, where ``dim`` is the
        size of the input dimension and ``vdim`` is the length of the return value
        of ``learner.function``."""
        return np.array([(*p, *np.atleast_1d(v)) for p, v in sorted(self.data.items())])

    def to_dataframe(
        self,
        with_default_function_args: bool = True,
        function_prefix: str = "function.",
        point_names: tuple[str, ...] = ("x", "y", "z"),
        value_name: str = "value",
    ) -> pandas.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Parameters
        ----------
        with_default_function_args : bool, optional
            Include the ``learner.function``'s default arguments as a
            column, by default True
        function_prefix : str, optional
            Prefix to the ``learner.function``'s default arguments' names,
            by default "function."
        point_names : tuple[str, ...], optional
            Names of the input points, should be the same length as number
            of input parameters by default ("x", "y", "z" )
        value_name : str, optional
            Name of the output value, by default "value"

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If `pandas` is not installed.
        """
        if not with_pandas:
            raise ImportError("pandas is not installed.")
        if len(point_names) != self.ndim:
            raise ValueError(
                f"point_names ({point_names}) should have the"
                f" same length as learner.ndims ({self.ndim})"
            )
        data = list((*x, y) for x, y in self.data.items())
        df = pandas.DataFrame(data, columns=[*point_names, value_name])
        df.attrs["inputs"] = list(point_names)
        df.attrs["output"] = value_name
        if with_default_function_args:
            assign_defaults(self.function, df, function_prefix)
        return df

    def load_dataframe(
        self,
        df: pandas.DataFrame,
        with_default_function_args: bool = True,
        function_prefix: str = "function.",
        point_names: tuple[str, ...] = ("x", "y", "z"),
        value_name: str = "value",
    ):
        """Load data from a `pandas.DataFrame`.

        If ``with_default_function_args`` is True, then ``learner.function``'s
        default arguments are set (using `functools.partial`) from the values
        in the `pandas.DataFrame`.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to load.
        with_default_function_args : bool, optional
            The ``with_default_function_args`` used in ``to_dataframe()``,
            by default True
        function_prefix : str, optional
            The ``function_prefix`` used in ``to_dataframe``, by default "function."
        point_names : str, optional
            The ``point_names`` used in ``to_dataframe``, by default ("x", "y", "z")
        value_name : str, optional
            The ``value_name`` used in ``to_dataframe``, by default "value"
        """
        self.tell_many(df[list(point_names)].values, df[value_name].values)
        if with_default_function_args:
            self.function = partial_function_from_dataframe(
                self.function, df, function_prefix
            )

    @property
    def bounds_are_done(self):
        return all(p in self.data for p in self._bounds_points)

    def _ip(self):
        """A `scipy.interpolate.LinearNDInterpolator` instance
        containing the learner's data."""
        # XXX: take our own triangulation into account when generating the _ip
        return interpolate.LinearNDInterpolator(self.points, self.values)

    @property
    def tri(self):
        """An `adaptive.learner.triangulation.Triangulation` instance
        with all the points of the learner."""
        if self._tri is not None:
            return self._tri

        try:
            self._tri = Triangulation(self.points)
        except ValueError:
            # A ValueError is raised if we do not have enough points or
            # the provided points are coplanar, so we need more points to
            # create a valid triangulation
            return None

        self._update_losses(set(), self._tri.simplices)
        return self._tri

    @property
    def values(self):
        """Get the values from `data` as a numpy array."""
        return np.array(list(self.data.values()), dtype=float)

    @property
    def points(self):
        """Get the points from `data` as a numpy array."""
        return np.array(list(self.data.keys()), dtype=float)

    def tell(self, point, value):
        point = tuple(point)

        if point in self.data:
            return  # we already know about the point

        if value is None:
            return self.tell_pending(point)

        self.pending_points.discard(point)
        tri = self.tri
        self.data[point] = value

        if not self.inside_bounds(point):
            return

        self._update_range(value)
        if tri is not None:
            simplex = self._pending_to_simplex.get(point)
            if simplex is not None and not self._simplex_exists(simplex):
                simplex = None
            to_delete, to_add = tri.add_point(point, simplex, transform=self._transform)
            self._update_losses(to_delete, to_add)

    def _simplex_exists(self, simplex):
        simplex = tuple(sorted(simplex))
        return simplex in self.tri.simplices

    def inside_bounds(self, point):
        """Check whether a point is inside the bounds."""
        if self._interior is not None:
            return self._interior.find_simplex(point, tol=1e-8) >= 0
        else:
            eps = 1e-8
            return all(
                (mn - eps) <= p <= (mx + eps) for p, (mn, mx) in zip(point, self._bbox)
            )

    def tell_pending(self, point, *, simplex=None):
        point = tuple(point)
        if not self.inside_bounds(point):
            return

        self.pending_points.add(point)

        if self.tri is None:
            return

        simplex = tuple(simplex or self.tri.locate_point(point))
        if not simplex:
            return
            # Simplex is None if pending point is outside the triangulation,
            # then you do not have subtriangles

        simplex = tuple(simplex)
        simplices = [self.tri.vertex_to_simplices[i] for i in simplex]
        neighbors = set.union(*simplices)
        # Neighbours also includes the simplex itself

        for simpl in neighbors:
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
            self._simplex_queue.add((subloss, simplex, subsimplex))

    def _ask_and_tell_pending(self, n=1):
        xs, losses = zip(*(self._ask() for _ in range(n)))
        return list(xs), list(losses)

    def ask(self, n, tell_pending=True):
        """Chose points for learners."""
        if not tell_pending:
            with restore(self):
                return self._ask_and_tell_pending(n)
        else:
            return self._ask_and_tell_pending(n)

    def _ask_bound_point(self):
        # get the next bound point that is still available
        new_point = next(
            p
            for p in self._bounds_points
            if p not in self.data and p not in self.pending_points
        )
        self.tell_pending(new_point)
        return new_point, np.inf

    def _ask_point_without_known_simplices(self):
        assert not self._bounds_available
        # pick a random point inside the bounds
        # XXX: change this into picking a point based on volume loss
        a = np.diff(self._bbox).flat
        b = np.array(self._bbox)[:, 0]
        p = None
        while p is None or not self.inside_bounds(p):
            r = np.array([self._random.random() for _ in range(self.ndim)])
            p = r * a + b
            p = tuple(p)

        self.tell_pending(p)
        return p, np.inf

    def _pop_highest_existing_simplex(self):
        # find the simplex with the highest loss, we do need to check that the
        # simplex hasn't been deleted yet
        while len(self._simplex_queue):
            # XXX: Need to add check that the loss is the most recent computed loss
            loss, simplex, subsimplex = self._simplex_queue.pop(0)
            if (
                subsimplex is None
                and simplex in self.tri.simplices
                and simplex not in self._subtriangulations
            ):
                return abs(loss), simplex, subsimplex
            if (
                simplex in self._subtriangulations
                and simplex in self.tri.simplices
                and subsimplex in self._subtriangulations[simplex].simplices
            ):
                return abs(loss), simplex, subsimplex

        # Could not find a simplex, this code should never be reached
        assert self.tri is not None
        raise AssertionError(
            "Could not find a simplex to subdivide. Yet there should always"
            "  be a simplex available if LearnerND.tri() is not None."
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

        point_new = tuple(choose_point_in_simplex(points, transform=self._transform))

        self._pending_to_simplex[point_new] = simplex
        self.tell_pending(point_new, simplex=simplex)  # O(??)

        return point_new, loss

    @property
    def _bounds_available(self):
        return any(
            (p not in self.pending_points and p not in self.data)
            for p in self._bounds_points
        )

    def _ask(self):
        if self._bounds_available:
            return self._ask_bound_point()  # O(1)

        if self.tri is None:
            # All bound points are pending or have been evaluated, but we do not
            # have enough evaluated points to construct a triangulation, so we
            # pick a random point
            return self._ask_point_without_known_simplices()  # O(1)

        return self._ask_best_point()  # O(log N)

    def _compute_loss(self, simplex):
        # get the loss
        vertices = self.tri.get_vertices(simplex)
        values = [self.data[tuple(v)] for v in vertices]

        # scale them to a cube with sides 1
        vertices = vertices @ self._transform
        values = self._output_multiplier * np.array(values)

        if self.nth_neighbors == 0:
            # compute the loss on the scaled simplex
            return float(
                self.loss_per_simplex(vertices, values, self._output_multiplier)
            )

        # We do need the neighbors
        neighbors = self.tri.get_opposing_vertices(simplex)

        neighbor_points = self.tri.get_vertices(neighbors)
        neighbor_values = [self.data.get(x, None) for x in neighbor_points]

        for i, point in enumerate(neighbor_points):
            if point is not None:
                neighbor_points[i] = point @ self._transform

        for i, value in enumerate(neighbor_values):
            if value is not None:
                neighbor_values[i] = self._output_multiplier * value

        return float(
            self.loss_per_simplex(
                vertices,
                values,
                self._output_multiplier,
                neighbor_points,
                neighbor_values,
            )
        )

    def _update_losses(self, to_delete: set, to_add: set):
        # XXX: add the points outside the triangulation to this as well
        pending_points_unbound = set()

        for simplex in to_delete:
            loss = self._losses.pop(simplex, None)
            subtri = self._subtriangulations.pop(simplex, None)
            if subtri is not None:
                pending_points_unbound.update(subtri.vertices)

        pending_points_unbound = {
            p for p in pending_points_unbound if p not in self.data
        }
        for simplex in to_add:
            loss = self._compute_loss(simplex)
            self._losses[simplex] = loss

            for p in pending_points_unbound:
                self._try_adding_pending_point_to_simplex(p, simplex)

            if simplex not in self._subtriangulations:
                self._simplex_queue.add((loss, simplex, None))
                continue

            self._update_subsimplex_losses(
                simplex, self._subtriangulations[simplex].simplices
            )

        if self.nth_neighbors:
            points_of_added_simplices = set.union(*[set(s) for s in to_add])
            neighbors = (
                self.tri.get_simplices_attached_to_points(points_of_added_simplices)
                - to_add
            )
            for simplex in neighbors:
                loss = self._compute_loss(simplex)
                self._losses[simplex] = loss

                if simplex not in self._subtriangulations:
                    self._simplex_queue.add((loss, simplex, None))
                    continue

                self._update_subsimplex_losses(
                    simplex, self._subtriangulations[simplex].simplices
                )

    def _recompute_all_losses(self):
        """Recompute all losses and pending losses."""
        # amortized O(N) complexity
        if self.tri is None:
            return

        # reset the _simplex_queue
        self._simplex_queue = SortedKeyList(key=_simplex_evaluation_priority)

        # recompute all losses
        for simplex in self.tri.simplices:
            loss = self._compute_loss(simplex)
            self._losses[simplex] = loss

            # now distribute it around the the children if they are present
            if simplex not in self._subtriangulations:
                self._simplex_queue.add((loss, simplex, None))
                continue

            self._update_subsimplex_losses(
                simplex, self._subtriangulations[simplex].simplices
            )

    @property
    def _scale(self):
        # get the output scale
        return self._max_value - self._min_value

    def _update_range(self, new_output):
        if self._min_value is None or self._max_value is None:
            # this is the first point, nothing to do, just set the range
            self._min_value = np.min(new_output)
            self._max_value = np.max(new_output)
            self._old_scale = self._scale or 1
            return False

        # if range in one or more directions is doubled, then update all losses
        self._min_value = min(self._min_value, np.min(new_output))
        self._max_value = max(self._max_value, np.max(new_output))

        scale_multiplier = 1 / (self._scale or 1)

        # the maximum absolute value that is in the range. Because this is the
        # largest number, this also has the largest absolute numerical error.
        max_absolute_value_in_range = max(abs(self._min_value), abs(self._max_value))
        # since a float has a relative error of 1e-15, the absolute error is the value * 1e-15
        abs_err = 1e-15 * max_absolute_value_in_range
        # when scaling the floats, the error gets increased.
        scaled_err = abs_err * scale_multiplier

        # do not scale along the axis if the numerical error gets too big
        if scaled_err > 1e-2:  # allowed_numerical_error = 1e-2
            scale_multiplier = 1

        self._output_multiplier = scale_multiplier

        scale_factor = self._scale / self._old_scale
        if scale_factor > self._recompute_losses_factor:
            self._old_scale = self._scale
            self._recompute_all_losses()
            return True
        return False

    @cache_latest
    def loss(self, real=True):
        # XXX: compute pending loss if real == False
        losses = self._losses if self.tri is not None else dict()
        return max(losses.values()) if losses else float("inf")

    def remove_unfinished(self):
        # XXX: implement this method
        self.pending_points = set()
        self._subtriangulations = dict()
        self._pending_to_simplex = dict()

    ##########################
    # Plotting related stuff #
    ##########################

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
            raise NotImplementedError(
                "holoviews currently does not support", "3D surface plots in bokeh."
            )
        if self.ndim != 2:
            raise NotImplementedError(
                "Only 2D plots are implemented: You can "
                "plot a 2D slice with 'plot_slice'."
            )
        x, y = self._bbox
        lbrt = x[0], y[0], x[1], y[1]

        if len(self.data) >= 4:
            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                scale_factor = np.product(np.diag(self._transform))
                a_sq = np.sqrt(np.min(self.tri.volumes()) * scale_factor)
                n = max(10, int(0.658 / a_sq))

            xs = ys = np.linspace(0, 1, n)
            xs = xs * (x[1] - x[0]) + x[0]
            ys = ys * (y[1] - y[0]) + y[0]
            z = self._ip()(xs[:, None], ys[None, :]).squeeze()

            im = hv.Image(np.rot90(z), bounds=lbrt)

            if tri_alpha:
                points = np.array(
                    [self.tri.get_vertices(s) for s in self.tri.simplices]
                )
                points = np.pad(
                    points[:, [0, 1, 2, 0], :],
                    pad_width=((0, 0), (0, 1), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                ).reshape(-1, 2)
                tris = hv.EdgePaths([points])
            else:
                tris = hv.EdgePaths([])
        else:
            im = hv.Image([], bounds=lbrt)
            tris = hv.EdgePaths([])

        im_opts = dict(cmap="viridis")
        tri_opts = dict(line_width=0.5, alpha=tri_alpha)
        no_hover = dict(plot=dict(inspection_policy=None, tools=[]))

        return im.opts(style=im_opts) * tris.opts(style=tri_opts, **no_hover)

    def plot_slice(self, cut_mapping, n=None):
        """Plot a 1D or 2D interpolated slice of a N-dimensional function.

        Parameters
        ----------
        cut_mapping : dict (int → float)
            for each fixed dimension the value, the other dimensions
            are interpolated. e.g. ``cut_mapping = {0: 1}``, so from
            dimension 0 ('x') to value 1.
        n : int
            the number of boxes in the interpolation grid along each axis
        """
        hv = ensure_holoviews()
        plot_dim = self.ndim - len(cut_mapping)
        if plot_dim == 1:
            if not self.data:
                return hv.Scatter([]) * hv.Path([])
            elif self.vdim > 1:
                raise NotImplementedError(
                    "multidimensional output not yet supported by `plot_slice`"
                )
            n = n or 201
            values = [
                cut_mapping.get(i, np.linspace(*self._bbox[i], n))
                for i in range(self.ndim)
            ]
            ind = next(i for i in range(self.ndim) if i not in cut_mapping)
            x = values[ind]
            y = self._ip()(*values)
            p = hv.Path((x, y))

            # Plot with 5% margins such that the boundary points are visible
            margin = 0.05 / self._transform[ind, ind]
            plot_bounds = (x.min() - margin, x.max() + margin)
            return p.redim(x=dict(range=plot_bounds))

        elif plot_dim == 2:
            if self.vdim > 1:
                raise NotImplementedError(
                    "holoviews currently does not support 3D surface plots in bokeh."
                )
            if n is None:
                # Calculate how many grid points are needed.
                # factor from A=√3/4 * a² (equilateral triangle)
                scale_factor = np.product(np.diag(self._transform))
                a_sq = np.sqrt(np.min(self.tri.volumes()) * scale_factor)
                n = max(10, int(0.658 / a_sq))

            xs = ys = np.linspace(0, 1, n)
            xys = [xs[:, None], ys[None, :]]
            values = [
                cut_mapping[i]
                if i in cut_mapping
                else xys.pop(0) * (b[1] - b[0]) + b[0]
                for i, b in enumerate(self._bbox)
            ]

            lbrt = [b for i, b in enumerate(self._bbox) if i not in cut_mapping]
            lbrt = np.reshape(lbrt, (2, 2)).T.flatten().tolist()

            if len(self.data) >= 4:
                z = self._ip()(*values).squeeze()
                im = hv.Image(np.rot90(z), bounds=lbrt)
            else:
                im = hv.Image([], bounds=lbrt)

            return im.opts(style=dict(cmap="viridis"))
        else:
            raise ValueError("Only 1 or 2-dimensional plots can be generated.")

    def plot_3D(self, with_triangulation=False, return_fig=False):
        """Plot the learner's data in 3D using plotly.

        Does *not* work with the
        `adaptive.notebook_integration.live_plot` functionality.

        Parameters
        ----------
        with_triangulation : bool, default: False
            Add the verticices to the plot.
        return_fig : bool, default: False
            Return the `plotly.graph_objs.Figure` object instead of showing
            the rendered plot (default).

        Returns
        -------
        plot : `plotly.offline.iplot` object
            The 3D plot of ``learner.data``.
        """
        plotly = ensure_plotly()

        plots = []

        vertices = self.tri.vertices
        if with_triangulation:
            Xe, Ye, Ze = [], [], []
            for simplex in self.tri.simplices:
                for s in itertools.combinations(simplex, 2):
                    Xe += [vertices[i][0] for i in s] + [None]
                    Ye += [vertices[i][1] for i in s] + [None]
                    Ze += [vertices[i][2] for i in s] + [None]

            plots.append(
                plotly.graph_objs.Scatter3d(
                    x=Xe,
                    y=Ye,
                    z=Ze,
                    mode="lines",
                    line=dict(color="rgb(125,125,125)", width=1),
                    hoverinfo="none",
                )
            )

        Xn, Yn, Zn = zip(*vertices)
        colors = [self.data[p] for p in self.tri.vertices]
        marker = dict(
            symbol="circle",
            size=3,
            color=colors,
            colorscale="Viridis",
            line=dict(color="rgb(50,50,50)", width=0.5),
        )

        plots.append(
            plotly.graph_objs.Scatter3d(
                x=Xn,
                y=Yn,
                z=Zn,
                mode="markers",
                name="actors",
                marker=marker,
                hoverinfo="text",
            )
        )

        axis = dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        )

        layout = plotly.graph_objs.Layout(
            showlegend=False,
            scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
            margin=dict(t=100),
            hovermode="closest",
        )

        fig = plotly.graph_objs.Figure(data=plots, layout=layout)

        return fig if return_fig else plotly.offline.iplot(fig)

    def _get_iso(self, level=0.0, which="surface"):
        if which == "surface":
            if self.ndim != 3 or self.vdim != 1:
                raise Exception(
                    "Isosurface plotting is only supported"
                    " for a 3D input and 1D output"
                )
            get_surface = True
            get_line = False
        elif which == "line":
            if self.ndim != 2 or self.vdim != 1:
                raise Exception(
                    "Isoline plotting is only supported for a 2D input and 1D output"
                )
            get_surface = False
            get_line = True

        vertices = []  # index -> (x,y,z)
        faces_or_lines = []  # tuple of indices of the corner points

        @functools.lru_cache()
        def _get_vertex_index(a, b):
            vertex_a = self.tri.vertices[a]
            vertex_b = self.tri.vertices[b]
            value_a = self.data[vertex_a]
            value_b = self.data[vertex_b]

            da = abs(value_a - level)
            db = abs(value_b - level)
            dab = da + db

            new_pt = db / dab * np.array(vertex_a) + da / dab * np.array(vertex_b)

            new_index = len(vertices)
            vertices.append(new_pt)
            return new_index

        for simplex in self.tri.simplices:
            plane_or_line = []
            for a, b in itertools.combinations(simplex, 2):
                va = self.data[self.tri.vertices[a]]
                vb = self.data[self.tri.vertices[b]]
                if min(va, vb) < level <= max(va, vb):
                    vi = _get_vertex_index(a, b)
                    should_add = True
                    for pi in plane_or_line:
                        if np.allclose(vertices[vi], vertices[pi]):
                            should_add = False
                    if should_add:
                        plane_or_line.append(vi)

            if get_surface and len(plane_or_line) == 3:
                faces_or_lines.append(plane_or_line)
            elif get_surface and len(plane_or_line) == 4:
                faces_or_lines.append(plane_or_line[:3])
                faces_or_lines.append(plane_or_line[1:])
            elif get_line and len(plane_or_line) == 2:
                faces_or_lines.append(plane_or_line)

        if len(faces_or_lines) == 0:
            r_min = min(self.data[v] for v in self.tri.vertices)
            r_max = max(self.data[v] for v in self.tri.vertices)

            raise ValueError(
                f"Could not draw isosurface for level={level}, as"
                " this value is not inside the function range. Please choose"
                f" a level strictly inside interval ({r_min}, {r_max})"
            )

        return vertices, faces_or_lines

    def plot_isoline(self, level=0.0, n=None, tri_alpha=0):
        """Plot the isoline at a specific level, only works in 2D.

        Parameters
        ----------
        level : float, default: 0
            The value of the function at which you would like to see
            the isoline.
        n : int
            The number of boxes in the interpolation grid along each axis.
            This is passed to `plot`.
        tri_alpha : float
            The opacity of the overlaying triangulation. This is passed
            to `plot`.

        Returns
        -------
        `holoviews.core.Overlay`
            The plot of the isoline(s). This overlays a `plot` with a
            `holoviews.element.Path`.
        """
        hv = ensure_holoviews()
        if n == -1:
            plot = hv.Path([])
        else:
            plot = self.plot(n=n, tri_alpha=tri_alpha)

        if isinstance(level, Iterable):
            for l in level:
                plot = plot * self.plot_isoline(level=l, n=-1)
            return plot

        vertices, lines = self._get_iso(level, which="line")
        paths = [[vertices[i], vertices[j]] for i, j in lines]
        contour = hv.Path(paths)

        contour_opts = dict(color="black")
        contour = contour.opts(style=contour_opts)
        return plot * contour

    def plot_isosurface(self, level=0.0, hull_opacity=0.2):
        """Plots a linearly interpolated isosurface.

        This is the 3D analog of an isoline. Does *not* work with the
        `adaptive.notebook_integration.live_plot` functionality.

        Parameters
        ----------
        level : float, default: 0.0
            the function value which you are interested in.
        hull_opacity : float, default: 0.0
            the opacity of the hull of the domain.

        Returns
        -------
        plot : `plotly.offline.iplot` object
            The plot object of the isosurface.
        """
        plotly = ensure_plotly()

        vertices, faces = self._get_iso(level, which="surface")
        x, y, z = zip(*vertices)

        fig = plotly.figure_factory.create_trisurf(
            x=x, y=y, z=z, plot_edges=False, simplices=faces, title="Isosurface"
        )
        isosurface = fig.data[0]
        isosurface.update(
            lighting=dict(ambient=1, diffuse=1, roughness=1, specular=0, fresnel=0)
        )

        if hull_opacity < 1e-3:
            # Do not compute the hull_mesh.
            return plotly.offline.iplot(fig)

        hull_mesh = self._get_hull_mesh(opacity=hull_opacity)
        return plotly.offline.iplot([isosurface, hull_mesh])

    def _get_hull_mesh(self, opacity=0.2):
        plotly = ensure_plotly()
        hull = scipy.spatial.ConvexHull(self._bounds_points)

        # Find the colors of each plane, giving triangles which are coplanar
        # the same color, such that a square face has the same color.
        color_dict = {}

        def _get_plane_color(simplex):
            simplex = tuple(simplex)
            # If the volume of the two triangles combined is zero then they
            # belong to the same plane.
            for simplex_key, color in color_dict.items():
                points = [hull.points[i] for i in set(simplex_key + simplex)]
                points = np.array(points)
                if np.linalg.matrix_rank(points[1:] - points[0]) < 3:
                    return color
                if scipy.spatial.ConvexHull(points).volume < 1e-5:
                    return color
            color_dict[simplex] = tuple(random.randint(0, 255) for _ in range(3))
            return color_dict[simplex]

        colors = [_get_plane_color(simplex) for simplex in hull.simplices]

        x, y, z = zip(*self._bounds_points)
        i, j, k = hull.simplices.T
        lighting = dict(ambient=1, diffuse=1, roughness=1, specular=0, fresnel=0)
        return plotly.graph_objs.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            facecolor=colors,
            opacity=opacity,
            lighting=lighting,
        )

    def _get_data(self):
        return deepcopy(self.__dict__)

    def _set_data(self, state):
        for k, v in state.items():
            setattr(self, k, v)
