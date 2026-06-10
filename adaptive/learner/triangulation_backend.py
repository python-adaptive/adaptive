"""Select the triangulation backend used by the learners.

If the optional Rust-accelerated `adaptive-triangulation
<https://github.com/python-adaptive/adaptive-triangulation>`_ package is
installed (``pip install "adaptive[rust]"``), it is used automatically as a
drop-in replacement for the pure-Python implementation in
`adaptive.learner.triangulation`, which makes `~adaptive.LearnerND`
significantly faster.

The selection can be overridden with the ``ADAPTIVE_TRIANGULATION_BACKEND``
environment variable:

- ``auto`` (default): use the Rust backend if available, else pure Python
- ``python``: always use the pure-Python implementation
- ``rust``: require the Rust backend, raising `ImportError` if it is missing

The active backend is exposed as the string ``TRIANGULATION_BACKEND``
(``"python"`` or ``"rust"``).

Note that the pure-Python implementation in `adaptive.learner.triangulation`
is always importable under its own name, regardless of the selected backend,
so pickles that reference it keep working.
"""

from __future__ import annotations

import os

# Minimal version that is a complete drop-in for the learners
# (incl. ``get_opposing_vertices`` and pickle/deepcopy support).
_MIN_RUST_VERSION = (0, 2, 1)


def _rust_version() -> tuple[int, ...] | None:
    """Return the installed ``adaptive_triangulation`` version, or None."""
    try:
        import adaptive_triangulation
    except ImportError:
        return None
    version = adaptive_triangulation.__version__
    return tuple(int(part) for part in version.split(".")[:3] if part.isdigit())


def _import_rust_triangulation():
    """Import the Rust `Triangulation`, raising a helpful `ImportError`."""
    version = _rust_version()
    if version is None:
        raise ImportError(
            "The 'rust' triangulation backend was requested but the "
            "'adaptive-triangulation' package is not installed. "
            'Install it with: pip install "adaptive[rust]"'
        )
    if version < _MIN_RUST_VERSION:
        raise ImportError(
            "The 'rust' triangulation backend requires "
            f"adaptive-triangulation >= {'.'.join(map(str, _MIN_RUST_VERSION))}, "
            f"found {'.'.join(map(str, version))}. Upgrade it with: "
            'pip install -U "adaptive[rust]"'
        )
    from adaptive_triangulation import Triangulation

    return Triangulation


def _select_backend() -> str:
    backend = os.environ.get("ADAPTIVE_TRIANGULATION_BACKEND", "auto").lower()
    if backend not in ("auto", "python", "rust"):
        raise ValueError(
            f"ADAPTIVE_TRIANGULATION_BACKEND={backend!r} is invalid, "
            "use 'auto', 'python', or 'rust'."
        )
    if backend == "auto":
        version = _rust_version()
        return (
            "rust" if version is not None and version >= _MIN_RUST_VERSION else "python"
        )
    if backend == "rust":
        _import_rust_triangulation()  # raise with guidance if unusable
    return backend


def resolve_triangulation_class(backend="auto"):
    """Return the `Triangulation` class to use for *backend*.

    Parameters
    ----------
    backend : str or type
        ``"auto"`` (the module-level default backend, which prefers the Rust
        implementation when available), ``"python"``, ``"rust"``, or a
        `Triangulation`-compatible class.
    """
    if isinstance(backend, type):
        return backend
    if backend == "auto":
        return Triangulation
    if backend == "python":
        from adaptive.learner.triangulation import Triangulation as tri_class

        return tri_class
    if backend == "rust":
        return _import_rust_triangulation()
    raise ValueError(
        f"Invalid triangulation backend {backend!r}, use 'auto', 'python', "
        "'rust', or a Triangulation-compatible class."
    )


TRIANGULATION_BACKEND: str = _select_backend()

if TRIANGULATION_BACKEND == "rust":
    from adaptive_triangulation import (
        Triangulation,
        circumsphere,
        fast_2d_circumcircle,
        fast_2d_point_in_simplex,
        fast_3d_circumcircle,
        fast_norm,
        orientation,
        point_in_simplex,
        simplex_volume_in_embedding,
    )
else:
    from adaptive.learner.triangulation import (
        Triangulation,
        circumsphere,
        fast_2d_circumcircle,
        fast_2d_point_in_simplex,
        fast_3d_circumcircle,
        fast_norm,
        orientation,
        point_in_simplex,
        simplex_volume_in_embedding,
    )

__all__ = [
    "TRIANGULATION_BACKEND",
    "Triangulation",
    "resolve_triangulation_class",
    "circumsphere",
    "fast_2d_circumcircle",
    "fast_2d_point_in_simplex",
    "fast_3d_circumcircle",
    "fast_norm",
    "orientation",
    "point_in_simplex",
    "simplex_volume_in_embedding",
]
