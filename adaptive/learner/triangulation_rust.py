"""
Rust-accelerated triangulation functions.

This module provides optimized implementations of core geometric functions
used by the triangulation module. Falls back to Python implementations if
the Rust extension is not available.
"""

import warnings

import numpy as np

try:
    from adaptive_rust import (
        circumsphere as rust_circumsphere,
    )
    from adaptive_rust import (
        fast_2d_point_in_simplex as rust_fast_2d_point_in_simplex,
    )
    from adaptive_rust import (
        fast_det as rust_fast_det,
    )
    from adaptive_rust import (
        fast_norm as rust_fast_norm,
    )
    from adaptive_rust import (
        point_in_simplex as rust_point_in_simplex,
    )
    from adaptive_rust import (
        simplex_volume_in_embedding as rust_simplex_volume_in_embedding,
    )
    from adaptive_rust import (
        volume as rust_volume,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    warnings.warn(
        "Rust extension not available, falling back to Python implementation. "
        "To enable Rust acceleration, build the extension with: maturin develop",
        RuntimeWarning,
        stacklevel=2,
    )

# Import Python fallbacks
from adaptive.learner.learnerND import volume as py_volume
from adaptive.learner.triangulation import (
    circumsphere as py_circumsphere,
)
from adaptive.learner.triangulation import (
    fast_2d_point_in_simplex as py_fast_2d_point_in_simplex,
)
from adaptive.learner.triangulation import (
    fast_det as py_fast_det,
)
from adaptive.learner.triangulation import (
    fast_norm as py_fast_norm,
)
from adaptive.learner.triangulation import (
    point_in_simplex as py_point_in_simplex,
)
from adaptive.learner.triangulation import (
    simplex_volume_in_embedding as py_simplex_volume_in_embedding,
)


def fast_det(matrix):
    """Fast determinant calculation."""
    if RUST_AVAILABLE:
        return rust_fast_det(np.asarray(matrix, dtype=float))
    return py_fast_det(matrix)


def fast_norm(v):
    """Fast vector norm calculation."""
    if RUST_AVAILABLE:
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return rust_fast_norm(v)
    return py_fast_norm(v)


def circumsphere(pts):
    """Calculate circumsphere of a simplex."""
    if RUST_AVAILABLE:
        pts_array = np.asarray(pts, dtype=float)
        center, radius = rust_circumsphere(pts_array)
        return tuple(center), radius
    return py_circumsphere(pts)


def point_in_simplex(point, simplex, eps=1e-8):
    """Check if a point is inside a simplex."""
    if RUST_AVAILABLE:
        # Convert to lists for Rust
        if isinstance(point, np.ndarray):
            point = point.tolist()
        if isinstance(simplex, np.ndarray):
            simplex = simplex.tolist()
        elif not isinstance(simplex, list):
            simplex = [list(s) if not isinstance(s, list) else s for s in simplex]
        return rust_point_in_simplex(point, simplex, eps)
    return py_point_in_simplex(point, simplex, eps)


def fast_2d_point_in_simplex(point, simplex, eps=1e-8):
    """Fast 2D point in triangle test."""
    if RUST_AVAILABLE:
        # Convert to tuples for Rust
        if isinstance(point, list | np.ndarray):
            point = tuple(point)
        if isinstance(simplex, np.ndarray):
            simplex = [tuple(s) for s in simplex]
        elif not all(isinstance(s, tuple) for s in simplex):
            simplex = [tuple(s) for s in simplex]
        return rust_fast_2d_point_in_simplex(point, simplex, eps)
    return py_fast_2d_point_in_simplex(point, simplex, eps)


def volume(simplex, ys=None):
    """Calculate volume of a simplex."""
    if RUST_AVAILABLE and ys is None:
        simplex_array = np.asarray(simplex, dtype=float)
        return rust_volume(simplex_array)
    return py_volume(simplex, ys)


def simplex_volume_in_embedding(vertices):
    """Calculate volume of a simplex in higher-dimensional embedding."""
    if RUST_AVAILABLE:
        vertices_array = np.asarray(vertices, dtype=float)
        return rust_simplex_volume_in_embedding(vertices_array)
    return py_simplex_volume_in_embedding(vertices)


def is_rust_available():
    """Check if Rust acceleration is available."""
    return RUST_AVAILABLE
