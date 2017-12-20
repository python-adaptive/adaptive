# -*- coding: utf-8 -*-
# Copyright 2010 Pedro Gonnet
# Copyright 2017 Christoph Groth
# Copyright 2017 `adaptive` authors

from collections import defaultdict
from math import sqrt
from operator import attrgetter

import numpy as np
from scipy.linalg import norm
from sortedcontainers import SortedSet

from .base_learner import BaseLearner
from .integrator_coeffs import (b_def, T_left, T_right, ns, hint,
                                ndiv_max, max_ivals, min_sep, eps,
                                xi, V_inv, Vcond, alpha, gamma)


def _downdate(c, nans, depth):
    # This is algorithm 5 from the thesis of Pedro Gonnet.
    b = b_def[depth].copy()
    m = ns[depth] - 1
    for i in nans:
        b[m + 1] /= alpha[m]
        xii = xi[depth][i]
        b[m] = (b[m] + xii * b[m + 1]) / alpha[m - 1]
        for j in range(m - 1, 0, -1):
            b[j] = ((b[j] + xii * b[j + 1] - gamma[j + 1] * b[j + 2])
                    / alpha[j - 1])
        b = b[1:]

        c[:m] -= c[m] / b[m] * b[:m]
        c[m] = 0
        m -= 1
    return c


def _zero_nans(fx):
    """Caution: this function modifies fx."""
    nans = []
    for i in range(len(fx)):
        if not np.isfinite(fx[i]):
            nans.append(i)
            fx[i] = 0.0
    return nans


def _calc_coeffs(fx, depth):
    """Caution: this function modifies fx."""
    nans = _zero_nans(fx)
    c_new = V_inv[depth] @ fx
    if nans:
        fx[nans] = np.nan
        c_new = _downdate(c_new, nans, depth)
    return c_new


class DivergentIntegralError(ValueError):
    pass


class _Interval:

    """
    Attributes
    ----------
    (a, b) : (float, float)
        The left and right boundary of the interval.
    c : numpy array of shape (4, 33)
        Coefficients of the fit.
    depth : int
        The level of refinement, `depth=0` means that it has 5 (the minimal
        number of) points and `depth=3` means it has 33 (the maximal number
        of) points.
    fx : numpy array of size `(5, 9, 17, 33)[self.depth]`.
        The function values at the points `self.points(self.depth)`.
    igral : float
        The integral value of the interval.
    err : float
        The error associated with the integral value.
    rdepth : int
        The number of splits that the interval has gone through, starting at 1.
    ndiv : int
        A number that is used to determine whether the interval is divergent.
    parent : _Interval
        The parent interval.
    children : list of `_Interval`s
        The intervals resulting from a split.
    done_points : dict
        A dictionary with the x-values and y-values: `{x1: y1, x2: y2 ...}`.
    done : bool
        The integral and the error for the interval has been calculated.
    done_leaves : set or None
        Leaves used for the error and the integral estimation of this
        interval. None means that this information was already propagated to
        the ancestors of this interval.
    depth_complete : int or None
        The level of refinement at which the interval has the integral value
        evaluated. If None there is no level at which the integral value is
        known yet.

    Methods
    -------
    refinement_complete : depth, optional
        If true, all the function values in the interval are known at `depth`.
        By default the depth is the depth of the interval.
    """

    __slots__ = [
        'a', 'b', 'c', 'c00', 'depth', 'igral', 'err', 'fx', 'rdepth',
        'ndiv', 'parent', 'children', 'done_points', 'done_leaves',
        'depth_complete'
    ]

    def __init__(self, a, b, depth, rdepth):
        self.children = []
        self.done_points = {}
        self.a = a
        self.b = b
        self.depth = depth
        self.rdepth = rdepth
        self.c00 = 0.0
        self.done_leaves = set()
        self.depth_complete = None

    @classmethod
    def make_first(cls, a, b, depth=2):
        ival = _Interval(a, b, depth, rdepth=1)
        ival.ndiv = 0
        ival.parent = None
        ival.err = np.inf
        return ival

    @property
    def T(self):
        """Get the correct shift matrix.

        Should only be called on children of a split interval.
        """
        assert self.parent is not None
        left = self.a == self.parent.a
        right = self.b == self.parent.b
        assert left != right
        return T_left if left else T_right

    def refinement_complete(self, depth):
        """The interval has all the y-values to calculate the intergral."""
        if len(self.done_points) < ns[depth]:
            return False
        return all(p in self.done_points for p in self.points(depth))

    def points(self, depth=None):
        if depth is None:
            depth = self.depth
        a = self.a
        b = self.b
        return (a + b) / 2 + (b - a) * xi[depth] / 2

    def refine(self):
        self.depth += 1
        return self

    def split(self):
        points = self.points()
        m = points[len(points) // 2]
        ivals = [_Interval(self.a, m, 0, self.rdepth + 1),
                 _Interval(m, self.b, 0, self.rdepth + 1)]
        self.children = ivals
        for ival in ivals:
            ival.parent = self
            ival.ndiv = self.ndiv
            ival.err = self.err / sqrt(2)

        return ivals

    def calc_igral(self):
        self.igral = (self.b - self.a) * self.c[0] / sqrt(2)

    def calc_err(self, c_old):
        c_new = self.c
        c_diff = np.zeros(max(len(c_old), len(c_new)))
        c_diff[:len(c_old)] = c_old
        c_diff[:len(c_new)] -= c_new
        c_diff = norm(c_diff)
        self.err = (self.b - self.a) * c_diff
        return c_diff

    def complete_process(self, depth):
        """Calculate the integral contribution and error from this interval,
        and update the done leaves of all ancestor intervals."""
        self.depth_complete = depth

        fx = [self.done_points[k] for k in self.points(depth)]
        self.fx = np.array(fx)
        first_process = self.parent is None and depth == 2
        if depth and not first_process:
            # Store for usage in refine
            c_old = self.c
        self.c = _calc_coeffs(self.fx, depth)

        if first_process:
            return False, False
        elif depth:
            # Refine
            c_diff = self.calc_err(c_old)
            force_split = c_diff > hint * norm(self.c)
        else:
            # Split
            parent = self.parent
            if parent.depth_complete is None:
                # Cannot reliably estimate the error, reverting to the heuristic
                N_up = 1
                while parent.parent is not None and parent.depth_complete is None:
                    parent = parent.parent
                    N_up += 1
                self.err = parent.err / 2**N_up

            c = parent.c if hasattr(parent, 'c') else np.zeros(33, dtype=float)

            c_old = self.T[:, :ns[parent.depth_complete]] @ c
            self.calc_err(c_old)
            self.c00 = self.c[0]

            self.ndiv = parent.ndiv + (parent.c00 and self.c00 / parent.c00 > 2)
            if self.ndiv > ndiv_max and 2*self.ndiv > self.rdepth:
                raise DivergentIntegralError(self)

            force_split = False

        self.calc_igral()

        if self.done_leaves is not None and not len(self.done_leaves):
            # This interval contributes to the integral estimate.
            self.done_leaves = {self}

            # Use this interval in the integral estimates of the ancestors
            # while possible.
            ival = self.parent
            old_leaves = set()
            while ival is not None:
                unused_children = [child for child in ival.children
                                   if child.done_leaves is not None]

                if not all(len(child.done_leaves) for child in unused_children):
                    break

                if ival.done_leaves is None:
                    ival.done_leaves = set()
                old_leaves.add(ival)
                for child in ival.children:
                    if child.done_leaves is None:
                        continue
                    ival.done_leaves.update(child.done_leaves)
                    child.done_leaves = None
                ival.done_leaves -= old_leaves
                ival = ival.parent

        # Check whether the point spacing is smaller than machine precision
        # and pop the interval with the largest error and do not split
        remove = self.err < (abs(self.igral) * eps * Vcond[depth])
        if remove:
            force_split = False

        return force_split, remove

    def __repr__(self):
        lst = [
            '(a, b)=({:.5f}, {:.5f})'.format(self.a, self.b),
            'depth={}'.format(self.depth),
            'rdepth={}'.format(self.rdepth),
            'err={:.5E}'.format(self.err),
            'igral={:.5E}'.format(self.igral if hasattr(self, 'igral') else np.inf),
        ]
        return ' '.join(lst)


class IntegratorLearner(BaseLearner):

    def __init__(self, function, bounds, tol):
        """
        Parameters
        ----------
        function : callable: X → Y
            The function to learn.
        bounds : pair of reals
            The bounds of the interval on which to learn 'function'.
        tol : float
            Relative tolerance of the error to the integral, this means that
            the learner is done when: `tol > err / abs(igral)`.

        Attributes
        ----------
        approximating_intervals : set of intervals
            The intervals that can be used in the determination of the integral.
        nr_points : int
            The total number of evaluated points.
        igral : float
            The integral value in `self.bounds`.
        err : float
            The absolute error associated with `self.igral`.

        Methods
        -------
        done : bool
            Returns whether the `tol` has been reached.
        plot : hv.Scatter
            Plots all the points that are evaluated.
        """
        self.function = function
        self.bounds = bounds
        self.tol = tol
        self.priority_split = []
        self.done_points = {}
        self.pending_points = set()
        self._stack = []
        self._err_excess = 0
        self._igral_excess = 0
        self.x_mapping = defaultdict(lambda: SortedSet([], key=attrgetter('rdepth')))
        self.ivals = SortedSet([], key=attrgetter('err'))
        ival = _Interval.make_first(*self.bounds)
        self._update_ival(ival)
        self.first_ival = ival

    @property
    def approximating_intervals(self):
        return self.first_ival.done_leaves

    def add_point(self, point, value):
        if point not in self.x_mapping:
            raise ValueError("Point {} doesn't belong to any interval"
                             .format(point))
        self.done_points[point] = value
        self.pending_points.discard(point)

        # Select the intervals that have this point
        ivals = self.x_mapping[point]
        for ival in ivals:
            force_split = False
            ival.done_points[point] = value

            if ival.depth_complete is None:
                from_depth = 2 if ival.parent is None else 0
            else:
                from_depth = ival.depth_complete + 1

            for depth in range(from_depth, ival.depth + 1):
                if ival.refinement_complete(depth):
                    in_ivals = ival in self.ivals
                    self.ivals.discard(ival)
                    force_split, remove = ival.complete_process(depth)
                    if remove:
                        # Remove the interval (while remembering the excess
                        # integral and error), since it is either too narrow,
                        # or the estimated relative error is already at the
                        # limit of numerical accuracy and cannot be reduced
                        # further.
                        self._err_excess += ival.err
                        self._igral_excess += ival.igral
                    elif in_ivals:
                        self.ivals.add(ival)

            if force_split:
                # Make sure that at the next execution of _fill_stack(),
                # this ival will be split.
                self.priority_split.append(ival)

    def _update_ival(self, ival):
        for x in ival.points():
            # Update the mappings
            self.x_mapping[x].add(ival)
            if x in self.done_points:
                self.add_point(x, self.done_points[x])
            elif x not in self.pending_points:
                self.pending_points.add(x)
                self._stack.append(x)

        # Add the new interval to the err sorted set
        self.ivals.add(ival)

    def choose_points(self, n):
        points, loss_improvements = self.pop_from_stack(n)
        n_left = n - len(points)
        while n_left > 0:
            assert n_left >= 0
            self._fill_stack()
            new_points, new_loss_improvements = self.pop_from_stack(n_left)
            points += new_points
            loss_improvements += new_loss_improvements
            n_left -= len(new_points)

        return points, loss_improvements

    def pop_from_stack(self, n):
        points = self._stack[:n]
        self._stack = self._stack[n:]
        loss_improvements = [max(ival.err for ival in self.x_mapping[x])
                             for x in points]
        return points, loss_improvements

    def remove_unfinished(self):
        pass

    def _fill_stack(self):
        # XXX: to-do if all the ivals have err=inf, take the interval
        # with the lowest rdepth and no children.
        if self.priority_split:
            ival = self.priority_split.pop()
            force_split = True
        else:
            ival = self.ivals[-1]
            force_split = False
            assert not ival.children

        # Remove the interval from the err sorted set because we're going to
        # split or refine this interval
        self.ivals.discard(ival)

        # If the interval points are smaller than machine precision, then
        # don't continue with splitting or refining.
        points = ival.points()

        reached_machine_tol = (points[1] - points[0] < points[0] * min_sep or
                               points[-1] - points[-2] < points[-2] * min_sep)

        if not reached_machine_tol:
            if ival.depth == 3 or force_split:
                # Always split when depth is maximal or if refining didn't help
                ivals_new = ival.split()
                for ival_new in ivals_new:
                    self._update_ival(ival_new)
            else:
                ival_new = ival.refine()
                self._update_ival(ival_new)

        # Remove the interval with the smallest error
        # if number of intervals is larger than max_ivals
        if len(self.ivals) > max_ivals:
            self.ivals.pop(0)

        return self._stack

    @property
    def nr_points(self):
        return len(self.done_points)

    @property
    def igral(self):
        return sum(i.igral for i in self.approximating_intervals)

    @property
    def err(self):
        if self.approximating_intervals:
            return sum(i.err for i in self.approximating_intervals)
        else:
            return np.inf

    def done(self):
        err = self.err
        igral = self.igral
        return (err == 0
                or err < abs(igral) * self.tol
                or (self._err_excess > abs(igral) * self.tol
                    and err - self._err_excess < abs(igral) * self.tol)
                or not self.ivals)

    def loss(self, real=True):
        return abs(abs(self.igral) * self.tol - self.err)

    def plot(self):
        import holoviews as hv
        return hv.Scatter(self.done_points)
