# -*- coding: utf-8 -*-

from collections import defaultdict
from copy import copy
from itertools import permutations

import numpy as np

from adaptive.learner.average_mixin import DataPoint, add_average_mixin
from adaptive.learner.learner2D import Learner2D


@add_average_mixin
class AverageLearner2D(Learner2D):
    """Same as 'Learner2D', only the differences are in the doc-string.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a tuple of a tuple of two real
        parameters and a seed and return a real number.
        So ((x, y), seed) → float, e.g.:
        >>> def f(xy_seed):
        ...     (x, y), seed = xy_seed
        ...     random.seed(xy_seed)
        ...     return x * y + random.uniform(-0.5, 0.5)
    average_priority : float, int, default 1
        When `average_priority > 1` adding more points to existing points
        will be prioritized (making the standard error of a point more
        imporant,) otherwise adding new triangles will be prioritized
        (making the loss of a triangle more important.)

    Attributes
    ----------
    min_seeds_per_point : int, default 3
        Minimum amount of values per point. This means that the
        standard error of a point is infinity until there are
        'min_seeds_per_point' for a point.

    Methods
    -------
    mean_values_per_point : callable
        Returns the average numbers of values per (x, y) value.

    Notes
    -----
    The total loss of the learner is still only determined by the
    max loss of the triangles.
    """

    def __init__(self, function, bounds, average_priority=1, loss_per_triangle=None):
        super().__init__(function, bounds, loss_per_triangle)
        self._data = dict()  # {point: {seed: value}} mapping
        self.pending_points = dict()  # {point: {seed}}

        # Adding a seed of 0 to the _stack to
        # make {((x, y), seed): loss_improvements, ...}.
        self._stack = {(p, 0): l for p, l in self._stack.items()}
        self.average_priority = average_priority
        self.min_seeds_per_point = 3
        self._seed_stack = []  # [(point, nseeds, loss_improvement), ...]

    def unpack_point(self, point):
        return tuple(point[0]), point[1]

    def value_scale(self):
        if self.data:
            _, values = self._data_in_bounds()
            z_scale = values.ptp()
            z_scale = z_scale if z_scale > 0 else 1
        else:
            z_scale = 1
        return z_scale

    def _ask_points_without_adding(self, n):
        points, loss_improvements = super().ask(n, tell_pending=False)
        return points, loss_improvements

    def _get_neighbor_mapping_new_points(self, points):
        tri = self.ip().tri
        simplices = {
            p: tri.simplices[tri.find_simplex(p_scaled)]
            for p, p_scaled in zip(points, self._scale(points))
        }

        points_unscaled = [tuple(p) for p in self._unscale(tri.points)]

        return {p: [points_unscaled[n] for n in ns] for p, ns in simplices.items()}

    def _get_neighbor_mapping_existing_points(self):
        tri = self.ip().tri
        _neighbors = defaultdict(set)
        for simplex in tri.vertices:
            for i, j in permutations(simplex, 2):
                _neighbors[i].add(j)

        neighbors = {}
        points = [tuple(p) for p in self._unscale(tri.points)]
        for k, v in _neighbors.items():
            neighbors[points[k]] = [points[i] for i in v]
        return neighbors

    def inside_bounds(self, xy_seed):
        xy, seed = self.unpack_point(xy_seed)
        return super().inside_bounds(xy)

    def _ensure_point(self, point):
        """Adding a point with seed = 0.
        This used in '_fill_stack' in the Learner2D."""
        return (tuple(point), 0)

    def remove_unfinished(self):
        self.pending_points = {}
        for p in self._bounds_points:
            if p not in self.data:
                self._stack[(p, 0)] = np.inf

    def plot_std_or_n(self, which="std"):
        """Plot the number of points or standard deviation.

        Parameters
        ----------
        which : str
            'n', 'std', 'mean', or 'standard_error'

        Returns
        -------
        plot : hv.Image
            Plot of the 'number of points' or 'std' per point.
        """
        assert which in ("n", "std", "standard_error", "mean")
        tmp_learner = Learner2D(None, bounds=self.bounds)
        tmp_learner._data = {k: getattr(v, which) for k, v in self._data.items()}
        return tmp_learner.plot().relabel(which)

    def tell(self, point, value):
        point = tuple(point)  # ((x, y), seed) tuple
        point_exists = point[0] in self._data
        self._add_to_data(point, value)
        if not self.inside_bounds(point):
            return
        self._remove_from_to_pending(point)
        if not point_exists:
            # we reset the interpolator cache
            self._ip = None
        self._stack.pop(point, None)

    def _set_data(self, data):
        # change dict -> DataPoint, because the points are saved using dicts
        self._data = {k: DataPoint(v) for k, v in data.items()}
        # Remove points from stack if they already exist
        for point, seed in copy(self._stack):
            if seed in self._data.get(point, set()):
                self._stack.pop((point, seed))
