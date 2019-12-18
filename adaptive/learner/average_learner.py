from math import sqrt
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from adaptive.learner.base_learner import BaseLearner
from adaptive.notebook_integration import ensure_holoviews
from adaptive.utils import cache_latest


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
        function: Callable,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        if atol is None and rtol is None:
            raise Exception("At least one of `atol` and `rtol` should be set.")
        if atol is None:
            atol = np.inf
        if rtol is None:
            rtol = np.inf

        self.data = {}
        self.pending_points = set()
        self.function = function
        self.atol = atol
        self.rtol = rtol
        self.npoints = 0
        self.sum_f = 0
        self.sum_f_sq = 0

    @property
    def n_requested(self) -> int:
        return self.npoints + len(self.pending_points)

    def ask(self, n: int, tell_pending: bool = True) -> Tuple[List[int], List[float]]:
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

    def tell(self, n: int, value: float) -> None:
        if n in self.data:
            # The point has already been added before.
            return

        self.data[n] = value
        self.pending_points.discard(n)
        self.sum_f += value
        self.sum_f_sq += value ** 2
        self.npoints += 1

    def tell_pending(self, n: int) -> None:
        self.pending_points.add(n)

    @property
    def mean(self) -> float:
        """The average of all values in `data`."""
        return self.sum_f / self.npoints

    @property
    def std(self) -> float:
        """The corrected sample standard deviation of the values
        in `data`."""
        n = self.npoints
        if n < 2:
            return np.inf
        numerator = self.sum_f_sq - n * self.mean ** 2
        if numerator < 0:
            # in this case the numerator ~ -1e-15
            return 0
        return sqrt(numerator / (n - 1))

    @cache_latest
    def loss(self, real: bool = True, *, n=None) -> float:
        if n is None:
            n = self.npoints if real else self.n_requested
        else:
            n = n
        if n < 2:
            return np.inf
        standard_error = self.std / sqrt(n)
        return max(
            standard_error / self.atol, standard_error / abs(self.mean) / self.rtol
        )

    def _loss_improvement(self, n: int) -> float:
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
        return hv.operation.histogram(vals, num_bins=num_bins, dimension=1)

    def _get_data(self) -> Tuple[Dict[int, float], int, float, float]:
        return (self.data, self.npoints, self.sum_f, self.sum_f_sq)

    def _set_data(self, data: Tuple[Dict[int, float], int, float, float]) -> None:
        self.data, self.npoints, self.sum_f, self.sum_f_sq = data
