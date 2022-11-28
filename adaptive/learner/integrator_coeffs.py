# Based on an adaptive quadrature algorithm by Pedro Gonnet
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache

import numpy as np
import scipy.linalg


def legendre(n: int) -> list[list[Fraction]]:
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


def scalar_product(a: list[Fraction], b: list[Fraction]) -> Fraction:
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


def calc_bdef(ns: tuple[int, int, int, int]) -> list[np.ndarray]:
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


def calc_V(x: np.ndarray, n: int) -> np.ndarray:
    V = [np.ones(x.shape), x.copy()]
    for i in range(2, n):
        V.append((2 * i - 1) / i * x * V[-1] - (i - 1) / i * V[-2])
    for i in range(n):
        V[i] *= np.sqrt(i + 0.5)
    return np.array(V).T


@lru_cache(maxsize=None)
def _coefficients():
    """Compute the coefficients on demand, in order to avoid doing linear algebra on import."""
    eps = np.spacing(1)

    # the nodes and Newton polynomials
    ns = (5, 9, 17, 33)
    xi = [-np.cos(np.linspace(0, np.pi, n)) for n in ns]

    # Make `xi` perfectly anti-symmetric, important for splitting the intervals
    xi = [(row - row[::-1]) / 2 for row in xi]

    # Compute the Vandermonde-like matrix and its inverse.
    V = [calc_V(x, n) for x, n in zip(xi, ns)]
    V_inv = list(map(scipy.linalg.inv, V))
    Vcond = [
        scipy.linalg.norm(a, 2) * scipy.linalg.norm(b, 2) for a, b in zip(V, V_inv)
    ]

    # Compute the shift matrices.
    T_left, T_right = (V_inv[3] @ calc_V((xi[3] + a) / 2, ns[3]) for a in [-1, 1])

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

    # set-up the downdate matrix
    k = np.arange(ns[3])
    alpha = np.sqrt((k + 1) ** 2 / (2 * k + 1) / (2 * k + 3))
    gamma = np.concatenate([[0, 0], np.sqrt(k[2:] ** 2 / (4 * k[2:] ** 2 - 1))])

    b_def = calc_bdef(ns)
    return locals()


def __getattr__(name):
    try:
        return _coefficients()[name]
    except KeyError:
        raise AttributeError(f"module {__name__} has no attribute {name}")
