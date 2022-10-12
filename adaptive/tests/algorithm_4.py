# Copyright 2010 Pedro Gonnet
# Copyright 2017 Christoph Groth

from collections import defaultdict
from fractions import Fraction
from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import inv, norm

eps = np.spacing(1)


def legendre(n: int) -> List[List[Fraction]]:
    """Return the first n Legendre polynomials.

    The polynomials have *standard* normalization, i.e.
    int_{-1}^1 dx L_n(x) L_m(x) = delta(m, n) * 2 / (2 * n + 1).

    The return value is a list of list of fraction.Fraction instances.
    """
    result = [[Fraction(1)], [Fraction(0), Fraction(1)]]
    if n <= 2:
        return result[:n]
    for i in range(2, n):
        # Use Bonnet's recursion formula.
        new = (i + 1) * [Fraction(0)]
        new[1:] = (r * (2 * i - 1) for r in result[-1])
        new[:-2] = (n - r * (i - 1) for n, r in zip(new[:-2], result[-2]))
        new[:] = (n / i for n in new)
        result.append(new)
    return result


def newton(n: int) -> np.ndarray:
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

    mod = 2 * (n - 1)
    terms = defaultdict(int)
    terms[0, 0] += 1

    for i in range(n):
        newterms = []
        for (d, a), m in terms.items():
            for b in [i, -i]:
                # In order to reduce the number of terms, cosine
                # arguments are mapped back to the inteval [0, pi/2).
                arg = (a + b) % mod
                if arg > n - 1:
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
    assert all(int(cfe) == ce for cfe, ce in zip(cf, c)), "Precision loss"

    cf /= 2.0 ** np.arange(n, -1, -1)
    return cf


def scalar_product(a: List[Fraction], b: List[Fraction]) -> Fraction:
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


def calc_bdef(ns: Tuple[int, int, int, int]) -> List[np.ndarray]:
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
        a = list(map(Fraction, newton(n)))
        for b in legs[: n + 1]:
            igral = scalar_product(a, b)

            # Normalize & store.  (The polynomials returned by
            # legendre() have standard normalization that is not
            # orthonormal.)
            poly.append(np.sqrt((2 * len(b) - 1) / 2) * igral)

        result.append(np.array(poly))
    return result


# Nodes and Newton polynomials.
n = (5, 9, 17, 33)
xi = [-np.cos(np.arange(n[j]) / (n[j] - 1) * np.pi) for j in range(4)]
# Make `xi` perfectly anti-symmetric, important for splitting the intervals
xi = [(row - row[::-1]) / 2 for row in xi]

b_def = calc_bdef(n)


def calc_V(xi: np.ndarray, n: int) -> np.ndarray:
    V = [np.ones(xi.shape), xi.copy()]
    for i in range(2, n):
        V.append((2 * i - 1) / i * xi * V[-1] - (i - 1) / i * V[-2])
    for i in range(n):
        V[i] *= np.sqrt(i + 0.5)
    return np.array(V).T


# Compute the Vandermonde-like matrix and its inverse.
V = [calc_V(*args) for args in zip(xi, n)]
V_inv = list(map(inv, V))
Vcond = [norm(a, 2) * norm(b, 2) for a, b in zip(V, V_inv)]

# Compute the shift matrices.
T_lr = [V_inv[3] @ calc_V((xi[3] + a) / 2, n[3]) for a in [-1, 1]]

# If the relative difference between two consecutive approximations is
# lower than this value, the error estimate is considered reliable.
# See section 6.2 of Pedro Gonnet's thesis.
hint = 0.1

# Smallest acceptable relative difference of points in a rule.  This was chosen
# such that no artifacts are apparent in plots of (i, log(a_i)), where a_i is
# the sequence of estimates of the integral value of an interval and all its
# ancestors..
min_sep = 16 * eps

ndiv_max = 20
max_ivals = 200
sqrt_one_half = np.sqrt(0.5)


k = np.arange(n[3])
alpha = np.sqrt((k + 1) ** 2 / (2 * k + 1) / (2 * k + 3))
gamma = np.concatenate([[0, 0], np.sqrt(k[2:] ** 2 / (4 * k[2:] ** 2 - 1))])


def _downdate(c: np.ndarray, nans: List[int], depth: int) -> None:
    # This is algorithm 5 from the thesis of Pedro Gonnet.
    b = b_def[depth].copy()
    m = n[depth] - 1
    for i in nans:
        b[m + 1] /= alpha[m]
        xii = xi[depth][i]
        b[m] = (b[m] + xii * b[m + 1]) / alpha[m - 1]
        for j in range(m - 1, 0, -1):
            b[j] = (b[j] + xii * b[j + 1] - gamma[j + 1] * b[j + 2]) / alpha[j - 1]
        b = b[1:]

        c[:m] -= c[m] / b[m] * b[:m]
        c[m] = 0
        m -= 1


def _zero_nans(fx: np.ndarray) -> List[int]:
    nans = []
    for i in range(len(fx)):
        if not np.isfinite(fx[i]):
            nans.append(i)
            fx[i] = 0.0
    return nans


def _calc_coeffs(fx: np.ndarray, depth: int) -> np.ndarray:
    """Caution: this function modifies fx."""
    nans = _zero_nans(fx)
    c_new = V_inv[depth] @ fx
    if nans:
        fx[nans] = np.nan
        _downdate(c_new, nans, depth)
    return c_new


class DivergentIntegralError(ValueError):
    def __init__(self, msg: str, igral: float, err: None, nr_points: int) -> None:
        self.igral = igral
        self.err = err
        self.nr_points = nr_points
        super().__init__(msg)


class _Interval:
    __slots__ = ["a", "b", "c", "fx", "igral", "err", "depth", "rdepth", "ndiv", "c00"]

    def __init__(
        self, a: Union[int, float], b: Union[int, float], depth: int, rdepth: int
    ) -> None:
        self.a = a
        self.b = b
        self.depth = depth
        self.rdepth = rdepth

    def points(self) -> np.ndarray:
        a = self.a
        b = self.b
        return (a + b) / 2 + (b - a) * xi[self.depth] / 2

    @classmethod
    def make_first(
        cls, f: Callable, a: int, b: int, depth: int = 2
    ) -> Tuple["_Interval", int]:
        ival = _Interval(a, b, depth, 1)
        fx = f(ival.points())
        ival.c = _calc_coeffs(fx, depth)
        ival.c00 = 0.0
        ival.fx = fx
        ival.ndiv = 0
        return ival, n[depth]

    def calc_igral_and_err(self, c_old: np.ndarray) -> float:
        self.c = c_new = _calc_coeffs(self.fx, self.depth)
        c_diff = np.zeros(max(len(c_old), len(c_new)))
        c_diff[: len(c_old)] = c_old
        c_diff[: len(c_new)] -= c_new
        c_diff = norm(c_diff)
        w = self.b - self.a
        self.igral = w * c_new[0] * sqrt_one_half
        self.err = w * c_diff
        return c_diff

    def split(
        self, f: Callable
    ) -> Union[Tuple[Tuple[float, float, float], int], Tuple[List["_Interval"], int]]:
        m = (self.a + self.b) / 2
        f_center = self.fx[(len(self.fx) - 1) // 2]

        rdepth = self.rdepth + 1
        ivals = [_Interval(self.a, m, 0, rdepth), _Interval(m, self.b, 0, rdepth)]
        points = np.concatenate([ival.points()[1:-1] for ival in ivals])
        nr_points = len(points)
        fxs = np.empty((2, n[0]))
        fxs[:, 0] = self.fx[0], f_center
        fxs[:, -1] = f_center, self.fx[-1]
        fxs[:, 1:-1] = f(points).reshape((2, -1))

        for ival, fx, T in zip(ivals, fxs, T_lr):
            ival.fx = fx
            ival.calc_igral_and_err(T[:, : self.c.shape[0]] @ self.c)

            ival.c00 = ival.c[0]
            ival.ndiv = self.ndiv + (self.c00 and ival.c00 / self.c00 > 2)
            if ival.ndiv > ndiv_max and 2 * ival.ndiv > ival.rdepth:
                # Signal a divergent integral.
                return (ival.a, ival.b, ival.b - ival.a), nr_points

        return ivals, nr_points

    def refine(self, f: Callable) -> Tuple[np.ndarray, bool, int]:
        """Increase degree of interval."""
        self.depth = depth = self.depth + 1
        points = self.points()
        fx = np.empty(n[depth])
        fx[0 : n[depth] : 2] = self.fx
        fx[1 : n[depth] - 1 : 2] = f(points[1 : n[depth] - 1 : 2])
        self.fx = fx
        split = self.calc_igral_and_err(self.c) > hint * norm(self.c)
        return points, split, n[depth] - n[depth - 1]


def algorithm_4(
    f: Callable, a: int, b: int, tol: float, N_loops: int = int(1e9)
) -> Tuple[float, float, int, List["_Interval"]]:
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

    ival, nr_points = _Interval.make_first(f, a, b)

    ivals = [ival]
    igral_excess = 0
    err_excess = 0
    i_max = 0

    for _ in range(N_loops):
        if ivals[i_max].depth == 3:
            split = True
        else:
            points, split, nr_points_inc = ivals[i_max].refine(f)
            nr_points += nr_points_inc

        if (
            points[1] - points[0] < points[0] * min_sep
            or points[-1] - points[-2] < points[-2] * min_sep
            or ivals[i_max].err
            < (abs(ivals[i_max].igral) * eps * Vcond[ivals[i_max].depth])
        ):
            # Remove the interval (while remembering the excess integral and
            # error), since it is either too narrow, or the estimated relative
            # error is already at the limit of numerical accuracy and cannot be
            # reduced further.
            err_excess += ivals[i_max].err
            igral_excess += ivals[i_max].igral
            ivals[i_max] = ivals[-1]
            ivals.pop()
        elif split:
            result, nr_points_inc = ivals[i_max].split(f)
            nr_points += nr_points_inc
            if isinstance(result, tuple):
                raise DivergentIntegralError(
                    "Possibly divergent integral in the interval"
                    " [{}, {}]! (h={})".format(*result),
                    ivals[i_max].igral * np.inf,
                    None,
                    nr_points,
                )
            ivals.extend(result)
            ivals[i_max] = ivals.pop()

        # Compute the total error and new max.
        i_max = 0
        i_min = 0
        err = err_excess
        igral = igral_excess
        for i in range(len(ivals)):
            if ivals[i].err > ivals[i_max].err:
                i_max = i
            elif ivals[i].err < ivals[i_min].err:
                i_min = i
            err += ivals[i].err
            igral += ivals[i].igral

        # If there are too many intervals, remove the one with smallest
        # contribution to the error.
        if len(ivals) > max_ivals:
            err_excess += ivals[i_min].err
            igral_excess += ivals[i_min].igral
            ivals[i_min] = ivals[-1]
            ivals.pop()
            if i_max == len(ivals):
                i_max = i_min

        if (
            err == 0
            or err < abs(igral) * tol
            or (err_excess > abs(igral) * tol and err - err_excess < abs(igral) * tol)
            or not ivals
        ):
            return igral, err, nr_points, ivals
    return igral, err, nr_points, ivals


# ############### Tests ################


def f0(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x * np.sin(1 / x) * np.sqrt(abs(1 - x))


def f7(x):
    return x**-0.5


def f24(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.floor(np.exp(x))


def f21(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    y = 0
    for i in range(1, 4):
        y += 1 / np.cosh(20**i * (x - 2 * i / 10))
    return y


def f63(
    x: Union[float, np.ndarray], alpha: float, beta: float
) -> Union[float, np.ndarray]:
    return abs(x - beta) ** alpha


def F63(x, alpha, beta):
    return (x - beta) * abs(x - beta) ** alpha / (alpha + 1)


def fdiv(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return abs(x - 0.987654321) ** -1.1


def f_one_with_nan(x):
    x = np.asarray(x)
    result = np.ones(x.shape)
    result[x == 0] = np.inf
    return result


def test_legendre():
    legs = legendre(11)
    comparisons = [
        (legs[0], [1], 1),
        (legs[1], [0, 1], 1),
        (legs[10], [-63, 0, 3465, 0, -30030, 0, 90090, 0, -109395, 0, 46189], 256),
    ]
    for a, b, div in comparisons:
        for c, d in zip(a, b):
            assert c * div == d


def test_scalar_product(n=33):
    legs = legendre(n)
    selection = [0, 5, 7, n - 1]
    for i in selection:
        for j in selection:
            assert scalar_product(legs[i], legs[j]) == (
                (i == j) and Fraction(2, 2 * i + 1)
            )


def simple_newton(n):
    """Slower than 'newton()' and prone to numerical error."""
    from itertools import combinations

    nodes = -np.cos(np.arange(n) / (n - 1) * np.pi)
    return [
        sum(np.prod(-np.asarray(sel)) for sel in combinations(nodes, n - d))
        for d in range(n + 1)
    ]


def test_newton():
    assert_allclose(newton(9), simple_newton(9), atol=1e-15)


def test_b_def(depth=1):
    legs = [np.array(leg, float) for leg in legendre(n[depth] + 1)]
    result = np.zeros(len(legs[-1]))
    for factor, leg in zip(b_def[depth], legs):
        factor *= np.sqrt((2 * len(leg) - 1) / 2)
        result[: len(leg)] += factor * leg
    assert_allclose(result, newton(n[depth]), rtol=1e-15)


def test_downdate(depth=3):
    fx = np.abs(xi[depth])
    fx[1::2] = np.nan
    c_downdated = _calc_coeffs(fx, depth)

    depth -= 1
    fx = np.abs(xi[depth])
    c = _calc_coeffs(fx, depth)

    assert_allclose(c_downdated[: len(c)], c, rtol=0, atol=1e-9)


def test_integration():
    old_settings = np.seterr(all="ignore")

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
        assert e.err is None
        assert e.nr_points == 431

    np.seterr(**old_settings)


def test_analytic(n=200):
    def f(x):
        return f63(x, alpha, beta)

    def F(x):
        return F63(x, alpha, beta)

    old_settings = np.seterr(all="ignore")

    np.random.seed(123)
    params = np.empty((n, 2))
    params[:, 0] = np.linspace(-0.5, -1.5, n)
    params[:, 1] = np.random.random_sample(n)

    false_negatives = 0
    false_positives = 0

    for alpha, beta in params:
        try:
            igral, err, nr_points = algorithm_4(f, 0, 1, 1e-3)
        except DivergentIntegralError:
            assert alpha < -0.8
            false_negatives += alpha > -1
        else:
            if alpha <= -1:
                false_positives += 1
            else:
                igral_exact = F(1) - F(0)
                assert alpha < -0.7 or abs(igral - igral_exact) < err

    assert false_negatives < 0.05 * n
    assert false_positives < 0.05 * n

    np.seterr(**old_settings)


if __name__ == "__main__":
    test_legendre()
    test_scalar_product()
    test_newton()
    test_b_def()
    test_downdate()
    test_integration()
    test_analytic()
