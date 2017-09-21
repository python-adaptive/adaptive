# Copyright 2010 Pedro Gonnet
# Copyright 2017 Christoph Groth

import warnings
import numpy as np
from numpy.linalg import cond
from scipy.linalg import inv, solve
from scipy.linalg.blas import dgemv

eps = np.spacing(1)

# The following two functions allow for almost bit-per-bit equivalence
# with the matlab code as interpreted by octave.

def norm(a):
    return np.sqrt(a @ a)


def mvmul(a, b):
    return dgemv(1.0, a, b)


# the nodes and newton polynomials
n = (5, 9, 17, 33)
xi = [-np.cos(np.arange(n[j])/(n[j]-1) * np.pi) for j in range(4)]
b_def = (np.array([0, .233284737407921723637836578544e-1,
                   0, -.831479419283098085685277496071e-1,
                   0, .0541462136776153483932540272848 ]),
         np.array([0, .883654308363339862264532494396e-4,
                   0, .238811521522368331303214066075e-3,
                   0, .135365534194038370983135068211e-2,
                   0, -.520710690438660595086839959882e-2,
                   0, .00341659266223572272892690737979 ]),
         np.array([0, .379785635776247894184454273159e-7,
                   0, .655473977795402040043020497901e-7,
                   0, .103479954638984842226816620692e-6,
                   0, .173700624961660596894381303819e-6,
                   0, .337719613424065357737699682062e-6,
                   0, .877423283550614343733565759649e-6,
                   0, .515657204371051131603503471028e-5,
                   0, -.203244736027387801432055290742e-4,
                   0, .0000134265158311651777460545854542 ]),
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

def calc_V(xi, n):
    V = [np.ones(xi.shape), xi.copy()]
    for i in range(2, n):
        V.append((2*i-1) / i * xi * V[-1] - (i-1) / i * V[-2])
    for i in range(n):
        V[i] *= np.sqrt(i + 0.5)
    return np.array(V).T

# compute the coefficients
V = [calc_V(*args) for args in zip(xi, n)]
V_inv = list(map(inv, V))
Vcond = list(map(cond, V))

# shift matrix
T_lr = [V_inv[3] @ calc_V((xi[3] + a) / 2, n[3]) for a in [-1, 1]]

# compute the integral
w = np.sqrt(0.5)                # legendre

# set-up the downdate matrix
k = np.arange(n[3])
U = (np.diag(np.sqrt((k+1)**2 / (2*k+1) / (2*k+3)))
     + np.diag(np.sqrt(k[2:]**2 / (4*k[2:]**2-1)), 2))


def _calc_coeffs(fx, depth):
    """Caution: this function modifies fx."""
    nans = []
    for i in range(len(fx)):
        if not np.isfinite(fx[i]):
            fx[i] = 0.0
            nans.append(i)
    c_new = mvmul(V_inv[depth], fx)
    if len(nans) > 0:
        b_new = b_def[depth].copy()
        n_new = n[depth] - 1
        for i in nans:
            b_new[:-1] = solve(
                (U[:n[depth], :n[depth]] - np.diag(np.ones(n[depth] - 1)
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


class _Interval:
    __slots__ = ['a', 'b', 'c', 'c_old', 'fx', 'igral', 'err', 'tol',
                 'rdepth', 'ndiv']

    @classmethod
    def make_first(cls, f, a, b, tol):
        points = (a+b)/2 + (b-a) * xi[3] / 2
        fx = f(points)
        nans = []
        for i in range(len(fx)):
            if not np.isfinite(fx[i]):
                nans.append(i)
                fx[i] = 0.0
        ival = _Interval()
        ival.c = np.zeros((4, n[3]))
        ival.c[3, :n[3]] = mvmul(V_inv[3], fx)
        ival.c[2, :n[2]] = mvmul(V_inv[2], fx[:n[3]:2])
        fx[nans] = np.nan
        ival.fx = fx
        ival.c_old = np.zeros(fx.shape)
        ival.a = a
        ival.b = b
        ival.igral = (b-a) * w * ival.c[3, 0]
        c_diff = norm(ival.c[3] - ival.c[2])
        ival.err = (b-a) * c_diff
        if c_diff / norm(ival.c[3]) > 0.1:
            ival.err = max( ival.err , (b-a) * norm(ival.c[3]) )
        ival.tol = tol
        ival.ndiv = 0
        ival.rdepth = 1
        return ival, points

    @property
    def depth(self):
        return n.index(len(self.fx)) + 1

    def split(self, f, ndiv_max=20):
        a = self.a
        b = self.b
        m = (a + b) / 2
        f_center = self.fx[(len(self.fx)-1)//2]

        ivals = []
        nr_points = 0
        for aa, bb, f_left, f_right, T in [
                (a, m, self.fx[0], f_center, T_lr[0]),
                (m, b, f_center, self.fx[-1], T_lr[1])]:
            ival = _Interval()
            ivals.append(ival)
            ival.a = aa
            ival.b = bb
            ival.tol = self.tol / np.sqrt(2)
            ival.rdepth = self.rdepth + 1
            ival.c = np.zeros((4, n[3]))
            fx = np.concatenate(
                ([f_left],
                 f((aa + bb) / 2 + (bb - aa) * xi[0][1:-1] / 2),
                 [f_right]))
            nr_points += n[0] - 2



            ival.c[0, :n[0]] = c_new = _calc_coeffs(fx, 0)
            ival.fx = fx

            ival.c_old = mvmul(T, self.c[self.depth - 1])
            c_diff = norm(ival.c[0] - ival.c_old)
            ival.err = (bb - aa) * c_diff
            ival.igral = (bb - aa) * ival.c[0, 0] * w
            ival.ndiv = (self.ndiv
                         + (abs(self.c[0, 0]) > 0
                            and ival.c[0, 0] / self.c[0, 0] > 2))
            if ival.ndiv > ndiv_max and 2*ival.ndiv > ival.rdepth:
                return (aa, bb, bb-aa), nr_points
        for ival in ivals:
            assert len(ival.fx) == n[ival.depth - 1]
        return ivals, nr_points

    def refine(self, f):
        """Increase degree of interval."""
        depth = self.depth
        if depth >= 4:
            raise RuntimeError('Max refinement level exceeded.')
        a = self.a
        b = self.b
        points = (a+b)/2 + (b-a)*xi[depth]/2
        fx = np.empty(n[depth])
        fx[0::2] = self.fx
        fx[1:-1:2] = f(points[1:n[depth]-1:2])
        fx = fx[:n[depth]]
        self.c[depth, :n[depth]] = c_new = _calc_coeffs(fx, depth)
        self.fx = fx
        c_diff = norm(self.c[depth - 1] - self.c[depth])
        self.err = (b-a) * c_diff
        self.igral = (b-a) * w * c_new[0]
        nc = norm(c_new)
        if nc > 0 and c_diff / nc > 0.1:
            split = True
        else:
            split = False
        assert len(self.fx) == n[self.depth - 1]
        return points, split, n[depth] - n[depth-1]


def algorithm_4 (f, a, b, tol):
    """ALGORITHM_4 evaluates an integral using adaptive quadrature. The
    algorithm uses Clenshaw-Curtis quadrature rules of increasing
    degree in each interval and bisects the interval if either the
    function does not appear to be smooth or a rule of maximum degree
    has been reached. The error estimate is computed from the L2-norm
    of the difference between two successive interpolations of the
    integrand over the nodes of the respective quadrature rules.

    INT = ALGORITHM_4 ( F , A , B , TOL ) approximates the integral of
    F in the interval [A,B] up to the relative tolerance TOL. The
    integrand F should accept a vector argument and return a vector
    result containing the integrand evaluated at each element of the
    argument.

    [INT,ERR,NR_POINTS] = ALGORITHM_4 ( F , A , B , TOL ) returns ERR,
    an estimate of the absolute integration error as well as
    NR_POINTS, the number of function values for which the integrand
    was evaluated. The value of ERR may be larger than the requested
    tolerance, indicating that the integration may have failed.

    ALGORITHM_4 halts with a warning if the integral is or appears to
    be divergent.

    Reference: "Increasing the Reliability of Adaptive Quadrature
        Using Explicit Interpolants", P. Gonnet, ACM Transactions on
        Mathematical Software, 37 (3), art. no. 26, 2008.
    """

    # compute the first interval
    ival, points = _Interval.make_first(f, a, b, tol)
    ivals = [ival]

    # init some globals
    igral = ival.igral
    err = ival.err
    igral_final = 0
    err_final = 0
    i_max = 0
    nr_points = n[3]

    # do we even need to go this way?
    if err < igral * tol:
        return igral, err, nr_points

    # main loop
    while True:
        if ivals[i_max].depth == 4:
            split = True
        else:
            points, split, nr_points_inc = ivals[i_max].refine(f)
            nr_points += nr_points_inc

        # can we safely ignore this interval?
        if (points[1] <= points[0]
            or points[-1] <= points[-2]
            or ivals[i_max].err < (abs(ivals[i_max].igral) * eps
                                   * Vcond[ivals[i_max].depth - 1])):
            err_final += ivals[i_max].err
            igral_final += ivals[i_max].igral
            ivals[i_max] = ivals.pop()
        elif split:
            result, nr_points_inc = ivals[i_max].split(f)
            nr_points += nr_points_inc
            if isinstance(result, tuple):
                igral = np.sign(igral) * np.inf
                raise DivergentIntegralError(
                    'Possibly divergent integral in the interval'
                    ' [{}, {}]! (h={})'.format(*result),
                    igral, err, nr_points)
            ivals.extend(result)
            ivals[i_max] = ivals.pop()

        # compute the running err and new max
        i_max = 0
        i_min = 0
        err = err_final
        igral = igral_final
        for i in range(len(ivals)):
            if ivals[i].err > ivals[i_max].err:
                i_max = i
            elif ivals[i].err < ivals[i_min].err:
                i_min = i
            err += ivals[i].err
            igral += ivals[i].igral

        # nuke smallest element if stack is larger than 200
        if len(ivals) > 200:
            err_final += ivals[i_min].err
            igral_final += ivals[i_min].igral
            ivals[i_min] = ivals.pop()
            if i_max == len(ivals):
                i_max = i_min

        # get up and leave?
        if (err == 0
            or err < abs(igral) * tol
            or (err_final > abs(igral) * tol
                and err - err_final < abs(igral) * tol)
            or not ivals):
            break

    return igral, err, nr_points


################ Tests ################

def f0(x):
    return x * np.sin(1/x) * np.sqrt(abs(1 - x))


def f7(x):
    return x**-0.5


def f24(x):
    return np.floor(np.exp(x))


def f21(x):
    y = 0
    for i in range(1, 4):
        y += 1 / np.cosh(20**i * (x - 2 * i / 10))
    return y


def f63(x):
    return abs(x - 0.987654321)**-0.45


def fdiv(x):
    return abs(x - 0.987654321)**-1.1


import struct

def float2hex(x):
    return struct.pack('!d', x).hex()

def hex2float(hex):
    return struct.unpack('!d', bytes.fromhex(hex))[0]

def assert_equal(value, hex, eps=0):
    assert (float2hex(value) == hex # for NaN, etc.
            or abs((value - hex2float(hex))) <= abs(eps * value)),\
            (abs((value - hex2float(hex))), eps, value)

def test():
    old_settings = np.seterr(all='ignore')

    igral, err, nr_points = algorithm_4(f0, 0, 3, 1e-5)
    print(igral, err, nr_points)
    assert_equal(igral, '3fffb6084c1dabf4')
    assert_equal(err, '3ef46042cb969374')
    assert nr_points == 1419, nr_points

    igral, err, nr_points = algorithm_4(f7, 0, 1, 1e-6)
    print(igral, err, nr_points)
    assert_equal(igral, '3fffffffd9fa6513')
    assert_equal(err, '3ebd8955755be30c')
    assert nr_points == 709, nr_points

    igral, err, nr_points = algorithm_4(f24, 0, 3, 1e-3)
    print(igral, err, nr_points)
    assert_equal(igral, '4031aa1505ba7b41')
    assert_equal(err, '3f9202232bd03a6a')
    assert nr_points == 4515, nr_points

    igral, err, nr_points = algorithm_4(f21, 0, 1, 1e-3)
    print(igral, err, nr_points)
    assert_equal(igral, '3fc4e088c36827c1')
    assert_equal(err, '3f247d00177a3f07')
    assert nr_points == 203, nr_points

    igral, err, nr_points = algorithm_4(f63, 0, 1, 1e-10)
    print(igral, err, nr_points)
    assert_equal(igral, '3fff7ccfd769d160')
    assert_equal(err, '3e28f421b487f15a', 2e-15)
    assert nr_points == 2715, nr_points

    try:
        igral, err, nr_points = algorithm_4(fdiv, 0, 1, 1e-6)
    except DivergentIntegralError as e:
        print(e.igral, e.err, e.nr_points)
        assert_equal(e.igral, '7ff0000000000000')
        assert_equal(e.err, '4073b48aeb356df5')
        assert e.nr_points == 457, e.nr_points

    np.seterr(**old_settings)


# if __name__ == '__main__':
#     test()


def __eq__(self, other):
    variables = []
    for slot in self.__slots__:
        try:
            eq = np.allclose(getattr(self, slot), getattr(other, slot), equal_nan=True)
        except:
            eq = getattr(self, slot) == getattr(other, slot)
        if not eq:
            print(slot, getattr(self, slot) - getattr(other, slot))
        variables.append(eq)
    return all(variables)


from math import sqrt
from copy import deepcopy as copy
from collections import defaultdict
import itertools
import operator
from sortedcontainers import SortedList, SortedDict, SortedSet
from adaptive.learner import BaseLearner

T_left, T_right = [V_inv[3] @ calc_V((xi[3] + a) / 2, n[3]) for a in [-1, 1]]

class Interval:
    __slots__ = ['a', 'b', 'c', 'c_old', 'depth', 'fx', 'igral', 'err', 'tol',
                 'rdepth', 'ndiv', 'parent', 'children', 'done_points', 'done']

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
        ival.c_old = np.zeros(n[ival.depth - 1])
        ival.err = np.inf
        ival.igral = 0
        return ival, ival.points(ival.depth - 1)

    @property
    def complete(self):
        return len(self.done_points) == ns[self.depth-1]

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

    def split_after_refine(self):
        depth = self.depth
        c_diff = norm(self.c[depth - 1] - self.c[depth])
        nc = norm(self.c[depth, :n[depth]])
        return nc > 0 and c_diff / nc > 0.1

    def split(self, f, ndiv_max=20):
        a = self.a
        b = self.b
        m = (a + b) / 2
        ivals = (Interval(a, m), Interval(m, b))
        ival_left, ival_right = ivals

        for ival in ivals:
            ival.depth = 1
            ival.tol = self.tol / np.sqrt(2)
            ival.rdepth = self.rdepth + 1
            ival.parent = self
            self.children.append(ival)
            ival.err = self.err / np.sqrt(2)  # XXX: is this correct even though we know some points?
            ival.igral = 0

        return ivals


class Learner(BaseLearner):
    def __init__(self, function, bounds, tol):
        self.function = function
        self.bounds = bounds
        self.tol = tol
        ival, points = Interval.make_first(*self.bounds, self.tol)

        self.ivals = SortedList([ival], key=operator.attrgetter('err'))
        self._stack = copy(list(points))
        self.x_mapping = defaultdict(lambda: SortedSet([], key=operator.attrgetter('rdepth')))
        for x in points:
            self.x_mapping[x].add(ival)

    def add_point(self, point, value):
        # Select the intervals that have this point
        ivals = self.x_mapping[point]
        while ivals:
            ival = ivals.pop(0)  # This pops the one with lowest rdepth
            ival.done_points[point] = value
            a, b = ival.a, ival.b
            if ival.complete:
                fx = np.array(ival.done_points.values())
                # Add points for the very first ival
                if ival.parent is None:
                    nans = []
                    for i in range(len(fx)):
                        if not np.isfinite(fx[i]):
                            nans.append(i)
                            fx[i] = 0.0

                    ival.c[3, :ns[3]] = V_inv[3] @ fx
                    ival.c[2, :ns[2]] = V_inv[2] @ fx[:ns[3]:2]
                    fx[nans] = np.nan
                    ival.fx = fx
                    ival.c_old = np.zeros(fx.shape)
                    c_diff = norm(ival.c[3] - ival.c[2])
                    ival.igral = (b - a) * ival.c[3, 0] / sqrt(2)
                    ival.err = (b - a) * c_diff

                    if c_diff / norm(ival.c[3]) > 0.1:
                        ival.err = max(ival.err, (b-a) * norm(ival.c[3]))
                else:
                    depth = ival.depth
                    if depth == 1:
                        # Split logic
                        ival.c[0, :n[0]] = c_new = _calc_coeffs(fx, 0)
                        ival.fx = fx
                        parent = ival.parent
                        ival.c_old = mvmul(ival.T, parent.c[parent.depth - 1])
                        c_diff = norm(ival.c[0] - ival.c_old)
                        ival.err = (b - a) * c_diff
                        ival.igral = (b - a) * ival.c[0, 0] / sqrt(2)
                        ival.ndiv = (parent.ndiv
                                     + (abs(parent.c[0, 0]) > 0
                                        and ival.c[0, 0] / parent.c[0, 0] > 2))
                        ndiv_max = 20
                        if ival.ndiv > ndiv_max and 2*ival.ndiv > ival.rdepth:
                            return (a, b, b-a), nr_points
                    else:
                        # Refine logic
                        ival.fx = fx
                        own_depth = ival.depth - 1
                        ival.c[own_depth, :n[own_depth]] = c_new = _calc_coeffs(fx, own_depth)
                        c_diff = norm(ival.c[own_depth - 1] - ival.c[own_depth])
                        ival.err = (b - a) * c_diff
                        ival.igral = (b - a) * c_new[0] / sqrt(2)
                    assert len(ival.fx) == n[ival.depth - 1]

    def choose_points(self, n):
        points, loss_improvements = self.take_from_stack(n)
        n_left = n - len(points)
        while n_left > 0:
            self._fill_stack()
            new_points, new_loss_improvements = self.take_from_stack(n_left)
            points += new_points
            loss_improvements += new_loss_improvements
            n_left -= len(new_points)

        return points, loss_improvements

    def take_from_stack(self, n):
        points = self._stack[:n]
        loss_improvements = [max(ival.err for ival in self.x_mapping[x])
                             for x in self._stack[:n]]

        # Remove from stack
        self._stack = self._stack[n:]
        return points, loss_improvements

    def loss(self, real=True):
        return self.err_final

    def remove_unfinished(self):
        pass

    def _fill_stack(self):
        # XXX: to-do if all the ivals have err=inf, take the interval
        # with the lowest rdepth and no children.
        ival = self.ivals[-1]
        points = ival.points(ival.depth - 1)

        if ival.depth == len(ns):
            # Always split when depth is maximal
            split = True
        elif ival.split_after_refine() and ival.complete:
            split = True
        else:
            # Refine
            points = ival.points(ival.depth)
            for x in points:
                self.x_mapping[x].add(ival)  # the values of x_mapping are sets
                if x in ival.done_points:
                    self.add_point(x, ival.done_points[x])
            self._stack += list(points[1::2])
            split = False
            ival.depth += 1

        # Check whether the point spacing is smaller than machine precision
        # and pop the interval with the largest error and do not split
        if (points[1] <= points[0]
            or points[-1] <= points[-2]
            or ival.err < (abs(ival.igral) * eps
                                   * Vcond[ival.depth - 1])):
            self.ivals.pop()
        elif split:
            ivals_new = ival.split(f)
            self.ivals.pop()

            done_points_parent = ival.done_points
            for ival in ivals_new:
                points = ival.points(depth=0)
                self._stack += list(points[1:-1])

                # Update the mappings
                # XXX: Do I remove the parent ival from the set?
                for x in points:
                    self.x_mapping[x].add(ival)

                for index in (0, -1):
                    # Add the known outermost points if they are done in the parent
                    x = points[index]
                    if x in done_points_parent:
                        self.add_point(x, done_points_parent[x])

            self.ivals += list(ivals_new)

        # Remove the smallest element if number of intervals is larger than 200
        if len(self.ivals) > 200:
            self.ivals.pop(0)

        return self._stack

    @property
    def nr_points(self):
        return sum(len(ival.done_points) for ival in self.ivals)

    @property
    def igral(self):
        return sum(ival.igral for ival in self.ivals
                   if ival.complete and not ival.children)

    @property
    def err(self):
        return sum(ival.err for ival in self.ivals
                   if ival.complete and not ival.children)

    def loss(self, real=True):
        return (err == 0
                or err < abs(igral) * tol
                or (err_final > abs(igral) * tol
                    and err - err_final < abs(igral) * tol)
                or not ivals)


f, a, b, tol = f0, 0, 3, 1e-5
l = Learner(f, bounds=(a, b), tol=tol)
points, loss_improvement = l.choose_points(33)
l.add_data(points, map(l.function, points))
ival_new = l.ivals[0]

ival, points = _Interval.make_first(f, a, b, tol)
ivals = [ival]
print(__eq__(ival, ival_new))
result, nr_points_inc = ivals.pop(0).split(f)
ivals.extend(result)

for i in range(6):
    print(i)
    points, loss_improvement = l.choose_points(1)
    l.add_data(points, map(l.function, points))

new1, new2 = l.ivals
old1, old2 = ivals
print(__eq__(old1, new1), __eq__(old2, new2))
