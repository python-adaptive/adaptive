from __future__ import annotations

from math import sqrt
from numbers import Integral as Int
from numbers import Real
from typing import Callable

import cloudpickle
import numpy as np

from adaptive.learner.base_learner import BaseLearner
from adaptive.notebook_integration import ensure_holoviews
from adaptive.types import Float
from adaptive.utils import (
    assign_defaults,
    cache_latest,
    partial_function_from_dataframe,
)

try:
    import pandas

    with_pandas = True

except ModuleNotFoundError:
    with_pandas = False


class AverageLearner(BaseLearner):
    """A naive implementation of adaptive computing of averages.

    The learned function must depend on an integer input variable that
    represents the source of randomness.

    Parameters
    ----------
    atol : float
        Desired absolute tolerance.
    rtol : float
        Desired relative tolerance.
    min_npoints : int
        Minimum number of points to sample.

    Attributes
    ----------
    data : dict
        Sampled points and values.
    pending_points : set
        Points that still have to be evaluated.
    npoints : int
        Number of evaluated points.
    """

    def __init__(
        self,
        function: Callable[[int], Real],
        atol: float | None = None,
        rtol: float | None = None,
        min_npoints: int = 2,
    ) -> None:
        if atol is None and rtol is None:
            raise Exception("At least one of `atol` and `rtol` should be set.")
        if atol is None:
            atol = np.inf
        if rtol is None:
            rtol = np.inf

        self.data = {}
        self.pending_points = set()
        self.function = function  # type: ignore
        self.atol = atol
        self.rtol = rtol
        self.npoints = 0
        # Cannot estimate standard deviation with fewer than 2 points.
        self.min_npoints = max(min_npoints, 2)
        self.sum_f: Real = 0.0
        self.sum_f_sq: Real = 0.0

    def new(self) -> AverageLearner:
        """Create a copy of `~adaptive.AverageLearner` without the data."""
        return AverageLearner(self.function, self.atol, self.rtol, self.min_npoints)

    @property
    def n_requested(self) -> int:
        return self.npoints + len(self.pending_points)

    def to_numpy(self):
        """Data as NumPy array of size (npoints, 2) with seeds and values."""
        return np.array(sorted(self.data.items()))

    def to_dataframe(
        self,
        with_default_function_args: bool = True,
        function_prefix: str = "function.",
        seed_name: str = "seed",
        y_name: str = "y",
    ) -> pandas.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Parameters
        ----------
        with_default_function_args : bool, optional
            Include the ``learner.function``'s default arguments as a
            column, by default True
        function_prefix : str, optional
            Prefix to the ``learner.function``'s default arguments' names,
            by default "function."
        seed_name : str, optional
            Name of the ``seed`` parameter, by default "seed"
        y_name : str, optional
            Name of the output value, by default "y"

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If `pandas` is not installed.
        """
        if not with_pandas:
            raise ImportError("pandas is not installed.")
        df = pandas.DataFrame(sorted(self.data.items()), columns=[seed_name, y_name])
        df.attrs["inputs"] = [seed_name]
        df.attrs["output"] = y_name
        if with_default_function_args:
            assign_defaults(self.function, df, function_prefix)
        return df

    def load_dataframe(
        self,
        df: pandas.DataFrame,
        with_default_function_args: bool = True,
        function_prefix: str = "function.",
        seed_name: str = "seed",
        y_name: str = "y",
    ):
        """Load data from a `pandas.DataFrame`.

        If ``with_default_function_args`` is True, then ``learner.function``'s
        default arguments are set (using `functools.partial`) from the values
        in the `pandas.DataFrame`.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to load.
        with_default_function_args : bool, optional
            The ``with_default_function_args`` used in ``to_dataframe()``,
            by default True
        function_prefix : str, optional
            The ``function_prefix`` used in ``to_dataframe``, by default "function."
        seed_name : str, optional
            The ``seed_name`` used in ``to_dataframe``, by default "seed"
        y_name : str, optional
            The ``y_name`` used in ``to_dataframe``, by default "y"
        """
        self.tell_many(df[seed_name].values, df[y_name].values)
        if with_default_function_args:
            self.function = partial_function_from_dataframe(
                self.function, df, function_prefix
            )

    def ask(self, n: int, tell_pending: bool = True) -> tuple[list[int], list[Float]]:
        points = list(range(self.n_requested, self.n_requested + n))

        if any(p in self.data or p in self.pending_points for p in points):
            # This means some of the points `< self.n_requested` do not exist.
            points = list(
                set(range(self.n_requested + n))
                - set(self.data)
                - set(self.pending_points)
            )[:n]

        loss_improvements = [self._loss_improvement(n) / n] * n
        if tell_pending:
            for p in points:
                self.tell_pending(p)
        return points, loss_improvements

    def tell(self, n: Int, value: Real) -> None:
        if n in self.data:
            # The point has already been added before.
            return

        self.data[n] = value
        self.pending_points.discard(n)
        self.sum_f += value
        self.sum_f_sq += value**2
        self.npoints += 1

    def tell_pending(self, n: int) -> None:
        self.pending_points.add(n)

    @property
    def mean(self) -> Float:
        """The average of all values in `data`."""
        return self.sum_f / self.npoints

    @property
    def std(self) -> Float:
        """The corrected sample standard deviation of the values
        in `data`."""
        n = self.npoints
        if n < self.min_npoints:
            return np.inf
        numerator = self.sum_f_sq - n * self.mean**2
        if numerator < 0:
            # in this case the numerator ~ -1e-15
            return 0
        return sqrt(numerator / (n - 1))

    @cache_latest
    def loss(self, real: bool = True, *, n=None) -> Float:
        if n is None:
            n = self.npoints if real else self.n_requested
        else:
            n = n
        if n < self.min_npoints:
            return np.inf
        standard_error = self.std / sqrt(n)
        aloss = standard_error / self.atol
        rloss = standard_error / self.rtol
        mean = self.mean
        if mean != 0:
            rloss /= abs(mean)
        return max(aloss, rloss)

    def _loss_improvement(self, n: int) -> Float:
        loss = self.loss()
        if np.isfinite(loss):
            return loss - self.loss(n=self.npoints + n)
        else:
            return np.inf

    def remove_unfinished(self):
        """Remove uncomputed data from the learner."""
        self.pending_points = set()

    def plot(self):
        """Returns a histogram of the evaluated data.

        Returns
        -------
        holoviews.element.Histogram
            A histogram of the evaluated data."""
        hv = ensure_holoviews()
        vals = [v for v in self.data.values() if v is not None]
        if not vals:
            return hv.Histogram([[], []])
        num_bins = int(max(5, sqrt(self.npoints)))
        vals = hv.Points(vals)
        return hv.operation.histogram(vals, num_bins=num_bins, dimension="y")

    def _get_data(self) -> tuple[dict[int, Real], int, Real, Real]:
        return (self.data, self.npoints, self.sum_f, self.sum_f_sq)

    def _set_data(self, data: tuple[dict[int, Real], int, Real, Real]) -> None:
        self.data, self.npoints, self.sum_f, self.sum_f_sq = data

    def __getstate__(self):
        return (
            cloudpickle.dumps(self.function),
            self.atol,
            self.rtol,
            self.min_npoints,
            self._get_data(),
        )

    def __setstate__(self, state):
        function, atol, rtol, min_npoints, data = state
        function = cloudpickle.loads(function)
        self.__init__(function, atol, rtol, min_npoints)
        self._set_data(data)
