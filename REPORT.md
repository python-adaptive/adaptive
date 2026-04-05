# LearnerND 1D Support — Python Implementation Report

## Summary

Added 1D support to `LearnerND` and the `Triangulation` class, allowing `LearnerND` to learn functions `f: R -> R^M` using the same adaptive triangulation infrastructure used for higher dimensions.

## Changes Made

### 1. `adaptive/learner/triangulation.py`

- **Removed `dim == 1` guard** (was at line 329): The `Triangulation.__init__` no longer raises `ValueError` for 1D points.

- **Added 1D initialization branch**: For `dim == 1`, sorts points by coordinate and creates interval simplices (pairs of consecutive sorted indices) instead of using `scipy.spatial.Delaunay` (which doesn't support 1D).

- **Fixed `simplex_volume_in_embedding`**: Added a `len(vertices) == 2` check before the Heron formula branch. For line segments (1-simplices), returns the Euclidean distance between the two vertices. This works correctly for line segments embedded in any dimension.

### 2. `adaptive/learner/learnerND.py`

- **Fixed `_ip()` for `ndim == 1`**: Uses `scipy.interpolate.interp1d` instead of `LinearNDInterpolator` (which requires ≥2 dimensions). Points are sorted before interpolation, with `bounds_error=False, fill_value=np.nan`.

- **Added 1D `plot()` support**: Before the `ndim != 2` guard, added a branch for `ndim == 1` that creates a holoviews `Path` for the interpolated curve and `Scatter` for known data points.

- **Updated error message**: Changed "Only 2D plots are implemented" to "Only 1D and 2D plots are implemented".

### 3. `adaptive/learner/data_saver.py`

- **Fixed `_to_key` for 1D tuple keys**: Added `use_tuple` parameter to handle LearnerND's tuple key convention (`(-1.0,)`) vs Learner1D's scalar keys (`-1.0`) when converting DataFrame rows back to dict keys.

- **Fixed `load_dataframe` indexing**: Changed `x[-1]` and `x[:-1]` to `x.iloc[-1]` and `x.iloc[:-1]` for correct positional indexing on pandas Series with string column labels.

### 4. Test Changes

#### `adaptive/tests/test_triangulation.py`
- Renamed `test_triangulation_raises_exception_for_1d_points` → `test_triangulation_supports_1d_points` (verifies 1D works instead of raising)
- Added `with_dimension_incl_1d` fixture (`[1, 2, 3, 4]`) for tests that are compatible with 1D
- Updated 8 parametrized tests to include `dim=1`
- Added 8 dedicated 1D tests: basic, multiple points, unsorted, add inside/outside left/right, locate point, opposing vertices

#### `adaptive/tests/unit/test_triangulation.py`
- Extended `test_circumsphere` range to start from `dim=1`
- Added `test_simplex_volume_in_embedding_1d` (1D, 2D, 3D embeddings)
- Added `test_1d_triangulation_find_simplices` and `test_1d_triangulation_find_neighbors`

#### `adaptive/tests/unit/test_learnernd.py`
- Added 6 tests: construction, tell/ask cycle, all loss functions, curvature loss, interpolation

#### `adaptive/tests/unit/test_learnernd_integration.py`
- Added 4 tests: run to N points (simple + blocking), curvature loss, loss-decreases integration test

#### `adaptive/tests/test_learnernd.py`
- Added 2 tests: basic 1D, 1D with loss goal

#### `adaptive/tests/test_learners.py`
- Registered `peak_1d` function with `@learn_with(LearnerND, bounds=((-1, 1),))` for cross-learner tests

## Why No Other Changes Were Needed

The existing `Triangulation` methods (add_point, bowyer_watson, _extend_hull, circumsphere, orientation, volume, hull, faces, etc.) all work correctly for 1D without modification because:

- **circumsphere**: The general N-dim formula correctly computes midpoint and half-length for 1D
- **point_in_simplex**: The linear algebra approach (`solve(vectors.T, ...)`) handles 1×1 systems
- **orientation**: `slogdet` of a 1×1 matrix correctly returns the sign
- **bowyer_watson**: Circumcircle-based point insertion naturally handles interval subdivision
- **hull**: Face counting correctly identifies endpoints of the 1D triangulation
- **volume**: `det` of a 1×1 matrix returns the interval length

## Test Results

All tests pass:

```
adaptive/tests/test_triangulation.py           — 68 passed
adaptive/tests/unit/test_triangulation.py      —  9 passed
adaptive/tests/unit/test_learnernd.py          —  9 passed
adaptive/tests/unit/test_learnernd_integration.py — 21 passed
adaptive/tests/test_learnernd.py               —  5 passed
adaptive/tests/test_learners.py (LearnerND)    — 52 passed
                                         Total: 164 passed
```

All existing 2D/3D/4D tests continue to pass unchanged.

## Verified Functionality

- All loss functions work for 1D: `default_loss`, `uniform_loss`, `std_loss`, `curvature_loss_function()`
- Interpolation produces correct values
- Loss decreases as more points are added
- `BlockingRunner` and `simple` runner both work
- DataFrame serialization/deserialization works (including DataSaver)
- Balancing learner works with 1D LearnerND
