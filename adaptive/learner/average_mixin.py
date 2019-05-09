# -*- coding: utf-8 -*-

import sys
from collections import defaultdict
from collections.abc import Mapping
from math import sqrt

import numpy as np
import scipy.stats

inf = sys.float_info.max


class AverageMixin:
    """The methods from this class are used in the
    `AverageLearner1D` and `AverageLearner2D.

    This cannot be used as a mixin class because of method resolution
    order problems. Instead use the `add_average_mixin` class decorator."""

    @property
    def data(self):
        return {k: v.mean for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {
            k: v.standard_error if v.n >= self.min_seeds_per_point else inf
            for k, v in self._data.items()
        }

    def mean_seeds_per_point(self):
        return np.mean([x.n for x in self._data.values()])

    def _next_seeds(self, point, nseeds, exclude=None):
        exclude = set(exclude) if exclude is not None else set()
        done_seeds = self._data.get(point, {}).keys()
        pending_seeds = self.pending_points.get(point, set())
        seeds = []
        for i in range(nseeds):
            seed = len(done_seeds) + len(pending_seeds) + len(exclude)
            if any(seed in x for x in [done_seeds, pending_seeds, exclude]):
                # Means that the seed already exists, for example
                # when 'done_seeds[point] | pending_seeds [point] == {0, 2}'.
                # Only happens when starting the learner after cancelling/loading.
                return list(
                    set(range(seed + nseeds - i)) - pending_seeds - done_seeds - exclude
                )[:nseeds]
            seeds.append(seed)
            exclude.add(seed)
        return seeds

    def _add_to_pending(self, point):
        x, seed = self.unpack_point(point)
        if x not in self.pending_points:
            self.pending_points[x] = set()
        self.pending_points[x].add(seed)

    def _remove_from_to_pending(self, point):
        x, seed = self.unpack_point(point)
        if x in self.pending_points:
            self.pending_points[x].discard(seed)
            if not self.pending_points[x]:
                # pending_points[x] is now empty so delete the set()
                del self.pending_points[x]

    def _add_to_data(self, point, value):
        x, seed = self.unpack_point(point)
        if x not in self._data:
            self._data[x] = DataPoint()
        self._data[x][seed] = value

    def ask(self, n, tell_pending=True):
        """Return n points that are expected to maximally reduce the loss."""
        points, loss_improvements = [], []

        # Take from the _seed_stack.
        self._fill_seed_stack(till=n)
        points = defaultdict(set)
        loss_improvements = {}
        remaining = n
        for i, (point, nseeds, loss_improvement) in enumerate(self._seed_stack):
            nseeds_to_choose = min(nseeds, remaining)
            seeds = self._next_seeds(point, nseeds_to_choose, exclude=points[point])
            for seed in seeds:
                points[point].add(seed)
                loss_improvements[(point, seed)] = loss_improvement / nseeds
            remaining -= nseeds_to_choose
            if not remaining:
                break

        # change from dict to list
        points = [(point, seed) for point, seeds in points.items() for seed in seeds]
        loss_improvements = [loss_improvements[point] for point in points]

        if tell_pending:
            # Remove the chosen points from the _seed_stack.
            for p in points:
                self.tell_pending(p)
            nseeds_left = nseeds - nseeds_to_choose  # of self._seed_stack[i]
            if nseeds_left > 0:  # not all seeds have been asked
                point, nseeds, loss_improvement = self._seed_stack[i]
                self._seed_stack[i] = (
                    point,
                    nseeds_left,
                    loss_improvement * nseeds_left / nseeds,
                )
                self._seed_stack = self._seed_stack[i:]
            else:
                self._seed_stack = self._seed_stack[i + 1 :]

        return points, loss_improvements

    def _fill_seed_stack(self, till):
        n = till - sum(nseeds for (_, nseeds, _) in self._seed_stack)
        if n <= 0:
            return

        new_points, new_interval_losses = self._interval_losses(n)
        existing_points, existing_points_sem_losses = self._point_losses()

        points = new_points + existing_points
        loss_improvements = new_interval_losses + existing_points_sem_losses

        priority = [
            (-loss / nseeds, loss, (point, nseeds))
            for loss, (point, nseeds) in zip(loss_improvements, points)
        ]

        _, loss_improvements, points = zip(*sorted(priority))

        # Add points to the _seed_stack, it can happen that its
        # length exceeds the number of requested points.
        n_left = n
        for loss_improvement, (point, nseeds) in zip(loss_improvements, points):
            self._seed_stack.append((point, nseeds, loss_improvement))
            n_left -= nseeds
            if n_left <= 0:
                break

    def n_values(self, point):
        """The total number of seeds (done or requested) at a point."""
        pending_points = self.pending_points.get(point, [])
        return len(self._data[point]) + len(pending_points)

    def _mean_seeds_per_neighbor(self, neighbors):
        """The average number of neighbors of a 'point'."""
        return {
            p: sum(self.n_values(n) for n in ns) / len(ns)
            for p, ns in neighbors.items()
        }

    def _interval_losses(self, n):
        """Add new points with at least self.min_seeds_per_point points
        or with as many points as the neighbors have on average."""
        points, loss_improvements = self._ask_points_without_adding(n)
        if len(self._data) < 4:  # ANTON: fix (4) to bounds
            points = [(p, self.min_seeds_per_point) for p, s in points]
            return points, loss_improvements

        only_points = [p for p, s in points]  # points are [(x, seed), ...]
        neighbors = self._get_neighbor_mapping_new_points(only_points)
        mean_seeds_per_neighbor = self._mean_seeds_per_neighbor(neighbors)

        points = []
        for p in only_points:
            n_neighbors = int(mean_seeds_per_neighbor[p])
            nseeds = max(n_neighbors, self.min_seeds_per_point)
            points.append((p, nseeds))

        return points, loss_improvements

    def _point_losses(self, fraction=1):
        """Increase the number of seeds by 'fraction'."""
        if len(self.data) < 4:
            return [], []
        scale = self.value_scale()
        points = []
        loss_improvements = []

        neighbors = self._get_neighbor_mapping_existing_points()
        mean_seeds_per_neighbor = self._mean_seeds_per_neighbor(neighbors)

        npoints_factor = np.log2(self.npoints)

        for p, sem in self.data_sem.items():
            N = self.n_values(p)
            n_more = int(fraction * N)  # increase the amount of points by fraction
            n_more = max(n_more, 1)  # at least 1 point
            points.append((p, n_more))
            needs_more_data = mean_seeds_per_neighbor[p] > 1.5 * N
            if needs_more_data:
                loss_improvement = inf
            else:
                # This is the improvement considering we will add
                # n_more seeds to the stack.
                sem_improvement = (1 - sqrt(N) / sqrt(N + n_more)) * sem
                # We scale the values, sem(ys) / scale == sem(ys / scale).
                # and multiply them by a weight average_priority.
                loss_improvement = self.average_priority * sem_improvement / scale
                if loss_improvement < inf:
                    loss_improvement *= npoints_factor
            loss_improvements.append(loss_improvement)
        return points, loss_improvements

    def _get_data(self):
        # change DataPoint -> dict for saving
        return {k: dict(v) for k, v in self._data.items()}


def add_average_mixin(cls):
    for name in AverageMixin.__dict__.keys():
        if not name.startswith("__") and not name.endswith("__"):
            # We assume that we don't implement or overwrite
            # dunder / magic methods inside AverageMixin.
            setattr(cls, name, getattr(AverageMixin, name))
    return cls


class DataPoint(dict):
    """A dict-like data structure that keeps track of the
    length, sum, and sum of squares of the values.

    It has properties to calculate the mean, sample
    standard deviation, and standard error."""

    def __init__(self, *args, **kwargs):
        self.sum = 0
        self.sum_sq = 0
        self.n = 0
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        self._remove(key)
        self.sum += val
        self.sum_sq += val ** 2
        self.n += 1
        super().__setitem__(key, val)

    def _remove(self, key):
        if key in self:
            val = self[key]
            self.sum -= val
            self.sum_sq -= val ** 2
            self.n -= 1

    @property
    def mean(self):
        return self.sum / self.n

    @property
    def std(self):
        if self.n < 2:
            # The sample standard deviation is not defined for
            # less than 2 values.
            return np.nan
        numerator = self.sum_sq - self.n * self.mean ** 2
        if numerator < 0:
            # This means that the numerator is ~ -1e-15
            # so nummerically it's 0.
            return 0
        return sqrt(numerator / (self.n - 1))

    @property
    def standard_error(self):
        if self.n < 2:
            return np.inf
        return self.std / sqrt(self.n)

    def __delitem__(self, key):
        self._remove(key)
        super().__delitem__(key)

    def pop(self, *args):
        self._remove(args[0])
        return super().pop(*args)

    def update(self, other=None, **kwargs):
        iterator = other if isinstance(other, Mapping) else kwargs
        for k, v in iterator.items():
            self[k] = v

    def assert_consistent_data_structure(self):
        vals = list(self.values())
        np.testing.assert_almost_equal(np.mean(vals), self.mean)
        np.testing.assert_almost_equal(np.std(vals, ddof=1), self.std)
        np.testing.assert_almost_equal(self.standard_error, scipy.stats.sem(vals))
        assert self.n == len(vals)
