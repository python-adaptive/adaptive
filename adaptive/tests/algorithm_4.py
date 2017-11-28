# Copyright 2010 Pedro Gonnet
# Copyright 2017 Christoph Groth

import warnings
from fractions import Fraction as Frac
from collections import defaultdict
import numpy as np
from numpy.testing import assert_allclose
from numpy.linalg import cond
from scipy.linalg import norm, inv


eps = np.spacing(1)

def legendre(n):
    """Return the first n Legendre polynomials.

    The polynomials have *standard* normalization, i.e.
    int_{-1}^1 dx L_n(x) L_m(x) = delta(m, n) * 2 / (2 * n + 1).

    The return value is a list of list of fraction.Fraction instances.
    """
    result = [[Frac(1)], [Frac(0), Frac(1)]]
    if n <= 2:
        return result[:n]
    for i in range(2, n):
        # Use Bonnet's recursion formula.
        new = (i + 1) * [Frac(0)]
        new[1:] = (r * (2*i - 1) for r in result[-1])
        new[:-2] = (n - r * (i - 1) for n, r in zip(new[:-2], result[-2]))
        new[:] = (n / i for n in new)
        result.append(new)
    return result


def newton(n):
    """Compute the monomial coefficients of the Newton polynomial over the
    nodes of the n-point Clenshaw-Curtis quadrature rule.
    """
    # The nodes of the Clenshaw-Curtis rule are x_i = -cos(i * Pi / (n-1)).
    # Here, we calculate the coefficients c_i such that sum_i c_i * x^i
    # = prod_i (x - x_i).  The coefficients are thus sums of products of
    # cosines.
    #
    # This routine uses the relation
    #   cos(a) cos(b) = (cos(a + b) + cos(a - b)) / 2
    # to efficiently calculate the coefficients.
    #
    # The dictionary 'terms' descibes the terms that make up the
    # monomial coefficients.  Each item ((d, a), m) corresponds to a
    # term m * cos(a * Pi / n) to be added to prefactor of the
    # monomial x^(n-d).

    mod = 2 * (n-1)
    terms = defaultdict(int)
    terms[0, 0] += 1

    for i in range(n):
        newterms = []
        for (d, a), m in terms.items():
            for b in [i, -i]:
                # In order to reduce the number of terms, cosine
                # arguments are mapped back to the inteval [0, pi/2).
                arg = (a + b) % mod
                if arg > n-1:
                    arg = mod - arg
                if arg >= n // 2:
                    if n % 2 and arg == n // 2:
                        # Zero term: ignore
                        continue
                    newterms.append((d + 1, n - 1 - arg, -m))
                else:
                    newterms.append((d + 1, arg, m))
        for d, s, m in newterms:
            terms[d, s] += m

    c = (n + 1) * [0]
    for (d, a), m in terms.items():
        if m and a != 0:
            raise ValueError("Newton polynomial cannot be represented exactly.")
        c[n - d] += m
        # The check could be removed and the above line replaced by
        # the following, but then the result would be no longer exact.
        # c[n - d] += m * np.cos(a * np.pi / (n - 1))

    cf = np.array(c, float)
    assert all(int(cfe) == ce for cfe, ce in zip(cf, c)), 'Precision loss'

    cf /= 2.**np.arange(n, -1, -1)
    return cf


def scalar_product(a, b):
    """Compute the polynomial scalar product int_-1^1 dx a(x) b(x).

    The args must be sequences of polynomial coefficients.  This
    function is careful to use the input data type for calculations.
    """
    la = len(a)
    lc = len(b) + la + 1

    # Compute the even coefficients of the product of a and b.
    c = lc * [a[0].__class__()]
    for i, bi in enumerate(b):
        if bi == 0:
            continue
        for j in range(i % 2, la, 2):
            c[i + j] += a[j] * bi

    # Calculate the definite integral from -1 to 1.
    return 2 * sum(c[i] / (i + 1) for i in range(0, lc, 2))


def calc_bdef(ns):
    """Calculate the decompositions of Newton polynomials (over the nodes
    of the n-point Clenshaw-Curtis quadrature rule) in terms of
    Legandre polynomials.

    The parameter 'ns' is a sequence of numers of points of the
    quadrature rule.  The return value is a corresponding sequence of
    normalized Legendre polynomial coefficients.
    """
    legs = legendre(max(ns) + 1)
    result = []
    for n in ns:
        poly = []
        a = list(map(Frac, newton(n)))
        for b in legs[:n + 1]:
            igral = scalar_product(a, b)

            # Normalize & store.  (The polynomials returned by
            # legendre() have standard normalization that is not
            # orthonormal.)
            poly.append(np.sqrt((2*len(b) - 1) / 2) * igral)

        result.append(np.array(poly))
    return result


# Nodes and Newton polynomials.
n = (5, 9, 17, 33)
xi = [-np.cos(np.pi / (n[j] - 1) * np.arange(n[j])) for j in range(4)]
# Set central rule points precisely to zero.  This does not really
# matter in practice, but is useful for tests.
for l in xi:
    l[len(l) // 2] = 0.0

b_def = calc_bdef(n)


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

# If the relative difference between two consecutive approximations is
# lower than this value, the error estimate is considered reliable.
# See section 6.2 of Pedro Gonnet's thesis.
hint = 0.1

# compute the integral
w = np.sqrt(0.5)                # legendre


k = np.arange(n[3])
alpha = np.sqrt((k+1)**2 / (2*k+1) / (2*k+3))
gamma = np.concatenate([[0, 0], np.sqrt(k[2:]**2 / (4*k[2:]**2-1))])

def _downdate(c, nans, depth):
    b = b_def[depth].copy()
    m = n[depth] - 1
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


def _zero_nans(fx):
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
        _downdate(c_new, nans, depth)
    return c_new


class DivergentIntegralError(ValueError):
    def __init__(self, msg, igral, err, nr_points):
        self.igral = igral
        self.err = err
        self.nr_points = nr_points
        super().__init__(msg)


class _Interval:
    __slots__ = ['a', 'b', 'c', 'fx', 'igral', 'err', 'depth', 'rdepth', 'ndiv']

    @classmethod
    def make_first(cls, f, a, b, depth=2):
        points = (a+b)/2 + (b-a) * xi[depth] / 2
        fx = f(points)
        ival = _Interval()
        ival.c = np.zeros((4, n[3]))
        ival.c[depth, :n[depth]] = _calc_coeffs(fx, depth)
        ival.fx = fx
        ival.a = a
        ival.b = b
        ival.depth = depth
        ival.ndiv = 0
        ival.rdepth = 1
        return ival, len(points)

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
            ival.depth = 0
            ival.rdepth = self.rdepth + 1
            ival.c = np.zeros((4, n[3]))
            fx = np.concatenate(
                ([f_left],
                 f((aa + bb) / 2 + (bb - aa) * xi[0][1:-1] / 2),
                 [f_right]))
            nr_points += n[0] - 2

            ival.c[0, :n[0]] = c_new = _calc_coeffs(fx, 0)
            ival.fx = fx

            c_old = T @ self.c[self.depth]
            c_diff = norm(ival.c[0] - c_old)
            ival.err = (bb - aa) * c_diff
            ival.igral = (bb - aa) * ival.c[0, 0] * w
            ival.ndiv = (self.ndiv
                         + (abs(self.c[0, 0]) > 0
                            and ival.c[0, 0] / self.c[0, 0] > 2))
            if ival.ndiv > ndiv_max and 2*ival.ndiv > ival.rdepth:
                return (aa, bb, bb-aa), nr_points

        return ivals, nr_points

    def refine(self, f):
        """Increase degree of interval."""
        self.depth = depth = self.depth + 1
        a = self.a
        b = self.b
        points = (a+b)/2 + (b-a)*xi[depth]/2
        fx = np.empty(n[depth])
        fx[0:n[depth]:2] = self.fx
        fx[1:n[depth]-1:2] = f(points[1:n[depth]-1:2])
        fx = fx[:n[depth]]
        self.c[depth, :n[depth]] = c_new = _calc_coeffs(fx, depth)
        self.fx = fx
        c_diff = norm(self.c[depth - 1] - self.c[depth])
        self.err = (b-a) * c_diff
        self.igral = (b-a) * w * c_new[0]
        nc = norm(c_new)
        split = nc > 0 and c_diff / nc > hint

        return points, split, n[depth] - n[depth - 1]


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

    # initialize the first interval
    ival, nr_points = _Interval.make_first(f, a, b)

    ivals = [ival]
    igral_final = 0
    err_final = 0
    i_max = 0

    # main loop
    while True:
        if ivals[i_max].depth == 3:
            split = True
        else:
            points, split, nr_points_inc = ivals[i_max].refine(f)
            nr_points += nr_points_inc

        # can we safely ignore this interval?
        if (points[1] <= points[0]
            or points[-1] <= points[-2]
            or ivals[i_max].err < (abs(ivals[i_max].igral) * eps
                                   * Vcond[ivals[i_max].depth])):
            err_final += ivals[i_max].err
            igral_final += ivals[i_max].igral
            ivals[i_max] = ivals[-1]
            ivals.pop()
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
            ivals[i_min] = ivals[-1]
            ivals.pop()
            if i_max == len(ivals):
                i_max = i_min

        # get up and leave?
        if (err == 0
            or err < abs(igral) * tol
            or (err_final > abs(igral) * tol
                and err - err_final < abs(igral) * tol)
            or not ivals):
            break

    return igral, err, nr_points, ivals


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


def f63(x, u, e):
    return abs(x-u)**e


def F63(x, u, e):
    return (x-u) * abs(x-u)**e / (e+1)


def fdiv(x):
    return abs(x - 0.987654321)**-1.1


def f_one_with_nan(x):
    x = np.asarray(x)
    result = np.ones(x.shape)
    result[x == 0] = np.inf
    return result


def test_legendre():
    legs = legendre(11)
    comparisons = [(legs[0], [1], 1),
                    (legs[1], [0, 1], 1),
                    (legs[10], [-63, 0, 3465, 0, -30030, 0,
                                90090, 0, -109395, 0, 46189], 256)]
    for a, b, div in comparisons:
        for c, d in zip(a, b):
            assert c * div == d


def test_scalar_product(n=33):
    legs = legendre(n)
    selection = [0, 5, 7, n-1]
    for i in selection:
        for j in selection:
            assert (scalar_product(legs[i], legs[j])
                    == ((i == j) and Frac(2, 2*i + 1)))


def simple_newton(n):
    """Slower than 'newton()' and prone to numerical error."""
    from itertools import combinations

    nodes = -np.cos(np.arange(n) / (n-1) * np.pi)
    return [sum(np.prod(-np.asarray(sel))
                for sel in combinations(nodes, n - d))
            for d in range(n + 1)]


def test_newton():
    assert_allclose(newton(9), simple_newton(9), atol=1e-15)


def test_b_def(depth=1):
    legs = [np.array(leg, float) for leg in legendre(n[depth] + 1)]
    result = np.zeros(len(legs[-1]))
    for factor, leg in zip(b_def[depth], legs):
        factor *= np.sqrt((2*len(leg) - 1) / 2)
        result[:len(leg)] += factor * leg
    assert_allclose(result, newton(n[depth]), rtol=1e-15)


def test_downdate(depth=3):
    fx = np.abs(xi[depth])
    fx[1::2] = np.nan
    c_downdated = _calc_coeffs(fx, depth)

    depth -= 1
    fx = np.abs(xi[depth])
    c = _calc_coeffs(fx, depth)

    assert_allclose(c_downdated[:len(c)], c, rtol=0, atol=1e-9)


def test_integration():
    old_settings = np.seterr(all='ignore')

    igral, err, nr_points = algorithm_4(f0, 0, 3, 1e-5)
    assert_allclose(igral, 1.98194117954329, 1e-15)
    assert_allclose(err, 1.9563545589988155e-05, 1e-10)
    assert nr_points == 1129

    igral, err, nr_points = algorithm_4(f7, 0, 1, 1e-6)
    assert_allclose(igral, 1.9999998579359648, 1e-15)
    assert_allclose(err, 1.8561437334964041e-06, 1e-10)
    assert nr_points == 693

    igral, err, nr_points = algorithm_4(f24, 0, 3, 1e-3)
    assert_allclose(igral, 17.664696186312934, 1e-15)
    assert_allclose(err, 0.017602618074957457, 1e-10)
    assert nr_points == 4519

    igral, err, nr_points = algorithm_4(f21, 0, 1, 1e-3)
    assert_allclose(igral, 0.16310022131213361, 1e-15)
    assert_allclose(err, 0.00011848806384952786, 1e-10)
    assert nr_points == 191

    igral, err, nr_points = algorithm_4(f_one_with_nan, -1, 1, 1e-12)
    assert_allclose(igral, 2, 1e-15)
    assert_allclose(err, 2.4237853822937613e-15, 1e-7)
    assert nr_points == 33

    try:
        igral, err, nr_points = algorithm_4(fdiv, 0, 1, 1e-6)
    except DivergentIntegralError as e:
        assert e.igral == np.inf
        assert_allclose(e.err, 284.56192231467958, 1e-10)
        assert e.nr_points == 431

    np.seterr(**old_settings)


def test_analytic(n=200):
    def f(x):
        return f63(x, u, e)

    def F(x):
        return F63(x, u, e)

    old_settings = np.seterr(all='ignore')

    np.random.seed(123)
    params = np.empty((n, 2))
    params[:, 0] = np.random.random_sample(n)
    params[:, 1] = np.linspace(-0.5, -1.5, n)

    false_negatives = 0
    false_positives = 0

    for u, e in params:
        try:
            igral, err, nr_points = algorithm_4(f, 0, 1, 1e-3)
        except DivergentIntegralError:
            assert e < -0.8
            false_negatives += e > -1
        else:
            if e <= -1:
                false_positives += 1
            else:
                igral_exact = F(1) - F(0)
                assert e < -0.8 or abs(igral - igral_exact) < err

    assert false_negatives < 0.05 * n
    assert false_positives < 0.05 * n

    np.seterr(**old_settings)


if __name__ == '__main__':
    test_legendre()
    test_scalar_product()
    test_newton()
    test_b_def()
    test_downdate()
    test_integration()
    test_analytic()
