# ISSUE-095 cleanup summary

This branch keeps the 1D `LearnerND` behavior from `issue-095-impl-py` and only
refactors the implementation and tests for clarity.

## What changed

### `refactor: consolidate 1D cleanup tests`

- Folded repeated 1D triangulation cases into parametrized tests.
- Reused small helpers in the new 1D `LearnerND` unit tests instead of repeating
  the same setup and tell/ask flow.
- Consolidated the 1D and 2D integration smoke tests so the shared runner
  behavior is checked in one place.

Why: the original 1D test additions covered the right behavior but repeated a
lot of setup and assertions, which made the cleanup surface harder to review.

### `refactor: clarify 1D triangulation and interpolation`

- Extracted `_flat_simplices()` so the 1D triangulation initialisation reads as
  one explicit helper instead of an inline special-case loop.
- Renamed local variables in `simplex_volume_in_embedding()` so the 2-vertex
  branch and the embedding-dimension branch are easier to read.
- Extracted `_sorted_line_data()` and `_plot_1d()` in `LearnerND` so the 1D
  interpolation and plotting paths are short, named, and isolated.
- Updated the `plot()` docstring to match the current 1D+2D behavior.

Why: the 1D support is still a real special case, but these helpers make the
special handling intentional and easier to follow without changing behavior.

### `refactor: simplify DataSaver key reconstruction`

- Replaced `_to_key(..., use_tuple=...)` with two small helpers:
  `_mapping_uses_tuple_keys()` and `_row_to_key()`.
- Switched `load_dataframe()` to inspect `self.learner.data` directly instead of
  relying on `__getattr__` indirection.
- Renamed `keys` to `input_columns` to match what the variable actually holds.

Why: the 1D `LearnerND` DataFrame fix depends on whether the wrapped learner uses
tuple keys for single inputs, and the new helper names make that rule explicit.

## Guardrails kept

- No public API changes.
- No behavior changes intended.
- Full `python -m pytest adaptive/tests/ -x --no-cov` runs were used to verify
  each cleanup slice before it was committed.
