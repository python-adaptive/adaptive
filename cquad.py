# Copyright 2010 Pedro Gonnet
# Copyright 2017 Christoph Groth
# Copyright 2017 `adaptive` authors

from collections import defaultdict
from copy import deepcopy as copy
from math import sqrt
from operator import attrgetter

import numpy as np
import scipy.linalg
from scipy.linalg.blas import dgemv
from sortedcontainers import SortedDict, SortedSet

from adaptive.learner import BaseLearner
eps = np.spacing(1)

# the nodes and newton polynomials
ns = (5, 9, 17, 33)
xi = [-np.cos(np.linspace(0, np.pi, n)) for n in ns]
# Make `xi` perfectly anti-symmetric, important for splitting the intervals
xi = [(row - row[::-1]) / 2 for row in xi]

# compute the coefficients
def calc_V(x, n):
    V = [np.ones(x.shape), x.copy()]
    for i in range(2, n):
        V.append((2*i-1) / i * x * V[-1] - (i-1) / i * V[-2])
    for i in range(n):
        V[i] *= np.sqrt(i + 0.5)
    return np.array(V).T

V = [calc_V(x, n) for x, n in zip(xi, ns)]
V_inv = list(map(scipy.linalg.inv, V))
Vcond = list(map(np.linalg.cond, V))

# shift matrix
T_left, T_right = [V_inv[3] @ calc_V((xi[3] + a) / 2, ns[3]) for a in [-1, 1]]

# set-up the downdate matrix
k = np.arange(ns[3])
U = (np.diag(np.sqrt((k+1)**2 / (2*k+1) / (2*k+3)))
     + np.diag(np.sqrt(k[2:]**2 / (4*k[2:]**2-1)), 2))


b_def = (np.array([0, .233284737407921723637836578544e-1,
                   0, -.831479419283098085685277496071e-1,
                   0, .0541462136776153483932540272848]),
         np.array([0, .883654308363339862264532494396e-4,
                   0, .238811521522368331303214066075e-3,
                   0, .135365534194038370983135068211e-2,
                   0, -.520710690438660595086839959882e-2,
                   0, .00341659266223572272892690737979]),
         np.array([0, .379785635776247894184454273159e-7,
                   0, .655473977795402040043020497901e-7,
                   0, .103479954638984842226816620692e-6,
                   0, .173700624961660596894381303819e-6,
                   0, .337719613424065357737699682062e-6,
                   0, .877423283550614343733565759649e-6,
                   0, .515657204371051131603503471028e-5,
                   0, -.203244736027387801432055290742e-4,
                   0, .0000134265158311651777460545854542]),
         np.array([0, .703046511513775683031092069125e-13,
                   0, .110617117381148770138566741591e-12,
                   0, .146334657087392356024202217074e-12,
                   0, .184948444492681259461791759092e-12,
                   0, .231429962470609662207589449428e-12,
                   0, .291520062115989014852816512412e-12,
                   0, .373653379768759953435020783965e-12,
                   0, .491840460397998449461993671859e-12,
                   0, .671514395653454630785723660045e-12,
                   0, .963162916392726862525650710866e-12,
                   0, .147853378943890691325323722031e-11,
                   0, .250420145651013003355273649380e-11,
                   0, .495516257435784759806147914867e-11,
                   0, .130927034711760871648068641267e-10,
                   0, .779528640561654357271364483150e-10,
                   0, -.309866395328070487426324760520e-9,
                   0, .205572320292667201732878773151e-9]))


def norm(a):
    return np.sqrt(a @ a)


def mvmul(a, b):
    return dgemv(1.0, a, b)


def _calc_coeffs(fx, depth):
    """Caution: this function modifies fx."""
    nans = []
    for i, f in enumerate(fx):
        if not np.isfinite(f):
            fx[i] = 0.0
            nans.append(i)
    c_new = mvmul(V_inv[depth], fx)
    if len(nans) > 0:
        b_new = b_def[depth].copy()
        n_new = ns[depth] - 1
        for i in nans:
            b_new[:-1] = scipy.linalg.solve(
                (U[:ns[depth], :ns[depth]] - np.diag(np.ones(ns[depth] - 1)
                                                     * xi[depth][i], 1)),
                b_new[1:])
            b_new[-1] = 0
            c_new -= c_new[n_new] / b_new[n_new] * b_new[:-1]
            n_new -= 1
            fx[i] = np.nan
    return c_new


class DivergentIntegralError(ValueError):
    def __init__(self, msg, igral, err, nr_points):
        self.igral = igral
        self.err = err
        self.nr_points = nr_points
        super().__init__(msg)


class Interval:
    __slots__ = ['a', 'b', 'c', 'c_old', 'depth', 'fx', 'igral', 'err', 'tol',
                 'rdepth', 'ndiv', 'parent', 'children', 'done_points']

    def __init__(self, a, b):
        self.children = []
        self.done_points = SortedDict()
        self.a = a
        self.b = b
        self.c = np.zeros((len(ns), ns[-1]))

    @classmethod
    def make_first(cls, a, b, tol):
        ival = Interval(a, b)
        ival.tol = tol
        ival.ndiv = 0
        ival.rdepth = 1
        ival.parent = None
        ival.depth = 4
        ival.c_old = np.zeros(ns[ival.depth - 1])
        ival.err = np.inf
        ival.igral = 0
        return ival, ival.points(ival.depth - 1)

    @property
    def complete(self):
        """The interval has all the y-values to calculate the intergral."""
        return len(self.done_points) == ns[self.depth - 1]

    @property
    def done(self):
        """The interval is complete and has the intergral calculated."""
        return hasattr(self, 'fx') and self.complete

    @property
    def T(self):
        if self.parent is not None:
            if self.a == self.parent.a:
                return T_left
            elif self.b == self.parent.b:
                return T_right
            else:
                raise Exception('This should not happen.')

    def points(self, depth):
        a = self.a
        b = self.b
        return (a+b)/2 + (b-a)*xi[depth]/2

    def refine(self):
        ival = Interval(self.a, self.b)
        ival.tol = self.tol
        ival.rdepth = self.rdepth
        ival.ndiv = self.ndiv
        ival.c = self.c.copy()
        ival.c_old = self.c_old.copy()
        ival.parent = self
        self.children.append(ival)
        ival.err = self.err
        ival.igral = 0
        points = ival.points(self.depth)
        ival.depth = self.depth + 1
        return ival, points

    def split(self):
        points = self.points(self.depth - 1)
        a = self.a
        b = self.b
        m = points[len(points) // 2]

        ivals = (Interval(a, m), Interval(m, b))
        ival_left, ival_right = ivals

        for ival in ivals:
            ival.depth = 1
            ival.tol = self.tol / sqrt(2)
            ival.c_old = self.c_old.copy()
            ival.rdepth = self.rdepth + 1
            ival.parent = self
            ival.ndiv = self.ndiv
            self.children.append(ival)
            ival.err = self.err / sqrt(2)
            ival.igral = 0

        return ivals

    def complete_process(self):
        force_split = False
        if self.parent is None:
            self.process_make_first()
        elif self.rdepth > self.parent.rdepth:
            self.process_split()
        else:
            force_split = self.process_refine()

        # Check whether the point spacing is smaller than machine precision
        # and pop the interval with the largest error and do not split
        remove = (abs(self.b - self.a) / ns[self.depth - 1] < eps
                  or self.err < (abs(self.igral) * eps
                                 * Vcond[self.depth - 1]))
        if remove:
            # If this interval is discarded from ivals, there is no need
            # to split it further.
            force_split = False

        return force_split, remove

    def process_make_first(self):
        fx = np.array(self.done_points.values())
        nans = []
        for i, f in enumerate(fx):
            if not np.isfinite(f):
                nans.append(i)
                fx[i] = 0.0

        self.c[3, :ns[3]] = V_inv[3] @ fx
        self.c[2, :ns[2]] = V_inv[2] @ fx[:ns[3]:2]
        fx[nans] = np.nan
        self.fx = fx
        self.c_old = np.zeros(fx.shape)
        c_diff = norm(self.c[3] - self.c[2])
        a, b = self.a, self.b
        self.igral = (b - a) * self.c[3, 0] / sqrt(2)
        self.err = (b - a) * c_diff

        if c_diff / norm(self.c[3]) > 0.1:
            self.err = max(self.err, (b-a) * norm(self.c[3]))

    def process_split(self, ndiv_max=20):
        fx = np.array(self.done_points.values())
        self.c[0, :ns[0]] = c_new = _calc_coeffs(fx, 0)
        self.fx = fx
        parent = self.parent

        self.c_old = mvmul(self.T, parent.c[parent.depth - 1])
        c_diff = norm(self.c[0] - self.c_old)

        a, b = self.a, self.b
        self.err = (b - a) * c_diff
        self.igral = (b - a) * self.c[0, 0] / sqrt(2)
        self.ndiv = (parent.ndiv
                     + (abs(parent.c[0, 0]) > 0
                        and self.c[0, 0] / parent.c[0, 0] > 2))
        if self.ndiv > ndiv_max and 2*self.ndiv > self.rdepth:
            return (a, b, b-a), nr_points

    def process_refine(self):
        fx = np.array(self.done_points.values())
        self.fx = fx
        own_depth = self.depth - 1
        self.c[own_depth, :ns[own_depth]] = c_new = _calc_coeffs(fx, own_depth)
        c_diff = norm(self.c[own_depth - 1] - self.c[own_depth])
        a, b = self.a, self.b
        self.err = (b - a) * c_diff
        self.igral = (b - a) * c_new[0] / sqrt(2)
        nc = norm(self.c[own_depth, :ns[own_depth]])
        force_split = nc > 0 and c_diff / nc > 0.1
        return force_split

    def __repr__(self):
        return str({'ab': (self.a, self.b), 'depth': self.depth,
                    'rdepth': self.rdepth, 'igral': self.igral, 'err': self.err})

    def equal(self, other, *, verbose=False):
        """Note: Implementing __eq__ breaks SortedContainers in some way."""
        if not self.complete:
            if verbose:
                print('Interval {} is not complete.'.format(self))
            return False

        slots = set(self.__slots__).intersection(other.__slots__)
        same_slots = []
        for s in slots:
            a = getattr(self, s)
            b = getattr(other, s)
            is_equal = np.allclose(a, b, rtol=0, atol=eps, equal_nan=True)
            if verbose and not is_equal:
                print('self.{} - other.{} = {}'.format(s, s, a - b))
            same_slots.append(is_equal)

        return all(same_slots)

class Learner(BaseLearner):
    def __init__(self, function, bounds, tol):
        self.function = function
        self.bounds = bounds
        self.tol = tol
        ival, points = Interval.make_first(*self.bounds, self.tol)
        self.priority_split = []
        self.ivals = SortedSet([ival], key=attrgetter('err'))
        self._stack = list(points)
        self._err_final = 0
        self._igral_final = 0
        self.x_mapping = defaultdict(lambda: SortedSet([], key=attrgetter('rdepth')))
        for x in points:
            self.x_mapping[x].add(ival)

    def add_point(self, point, value):
        # Select the intervals that have this point
        ivals = self.x_mapping[point]
        for ival in ivals:
            ival.done_points[point] = value
            if ival.complete and not ival.done:
                self.ivals.discard(ival)
                force_split, remove = ival.complete_process()
                if remove:
                    self._err_final += ival.err
                    self._igral_final += ival.igral
                if not remove:
                    self.ivals.add(ival)

                if force_split:
                    # Make sure that at the next execution of _fill_stack(), this ival will be split
                    self.priority_split.append(ival)

    def choose_points(self, n):
        points, loss_improvements = self.pop_from_stack(n)
        n_left = n - len(points)
        while n_left > 0:
            self._fill_stack()
            new_points, new_loss_improvements = self.pop_from_stack(n_left)
            points += new_points
            loss_improvements += new_loss_improvements
            n_left -= len(new_points)

        return points, loss_improvements

    def pop_from_stack(self, n):
        points = self._stack[:n]
        loss_improvements = [max(ival.err for ival in self.x_mapping[x])
                             for x in self._stack[:n]]

        # Remove from stack
        self._stack = self._stack[n:]
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

        points = ival.points(ival.depth - 1)

        if ival.depth == len(ns) or force_split:
            # Always split when depth is maximal or if refining is not helping
            split = True
        else:
            # Refine
            self.ivals.remove(ival)
            ival_new, points = ival.refine()
            for x in points:
                self.x_mapping[x].add(ival_new)
                if x in ival.done_points:
                    self.add_point(x, ival.done_points[x])
            self.ivals.add(ival_new)
            self._stack += list(points[1::2])
            split = False

        if split:
            self.ivals.remove(ival)

            if force_split:
                ival = copy(ival)
                ival.depth -= 1
            ivals_new = ival.split()

            done_points_parent = ival.done_points
            for ival in ivals_new:
                points = ival.points(depth=0)
                self._stack += list(points[1:-1])

                # Update the mappings
                for x in points:
                    self.x_mapping[x].add(ival)

                # Add the known outermost points if they are done in the parent
                for index in (0, -1):
                    x = points[index]
                    if x in done_points_parent:
                        self.add_point(x, done_points_parent[x])

            # Add the new intervals to the err sorted set
            self.ivals.update(ivals_new)

        # Remove the smallest element if number of intervals is larger than 200
        if len(self.ivals) > 200:
            print('nuke')
            self.ivals.pop(0)

        return self._stack

    @property
    def nr_points(self):
        return sum(1 for x, ivals in self.x_mapping.items()
                   if any(ival.complete for ival in ivals))

    @property
    def igral(self):
        # XXX: Need some recursion here for the parallel execution.
        # When `not ival.complete` take the `i.igral for i in ival.children`.
        return self._igral_final + sum(ival.igral for ival in self.ivals
                                       if ival.complete and not ival.children)

    @property
    def err(self):
        return self._err_final + sum(ival.err for ival in self.ivals
                                     if ival.complete and not ival.children)

    @property
    def first_ival(self):
        def go_up(ival):
            return go_up(ival.parent) if ival.parent else ival
        return go_up(self.ivals[0])

    def loss(self, real=True):
        return (self.err == 0
                or self.err < abs(self.igral) * self.tol
                or (self._err_final > abs(self.igral) * self.tol
                    and self.err - self._err_final < abs(self.igral) * self.tol)
                or not self.ivals)

    def equal(self, other, *, verbose=False):
        """Note: `other` is a list of ivals."""
        if len(self.ivals) != len(other):
            if verbose:
                print('len(self.ivals)={} != len(other)={}'.format(
                    len(self.ivals), len(other)))
            return False

        ivals = [sorted(i, key=attrgetter('a')) for i in [self.ivals, other]]
        return all(ival.equal(other_ival, verbose=verbose)
                   for ival, other_ival in zip(*ivals))
