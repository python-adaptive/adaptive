from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np
from skopt import Optimizer

from adaptive.learner.base_learner import BaseLearner
from adaptive.notebook_integration import ensure_holoviews
from adaptive.utils import cache_latest

if TYPE_CHECKING:
    import pandas


class SKOptLearner(Optimizer, BaseLearner):
    """Learn a function minimum using ``skopt.Optimizer``.

    This is an ``Optimizer`` from ``scikit-optimize``,
    with the necessary methods added to make it conform
    to the ``adaptive`` learner interface.

    Parameters
    ----------
    function : callable
        The function to learn.
    **kwargs :
        Arguments to pass to ``skopt.Optimizer``.
    """

    def __init__(self, function, **kwargs):
        self.function = function
        self.pending_points = set()
        self.data = collections.OrderedDict()
        self._kwargs = kwargs
        super().__init__(**kwargs)

    def new(self) -> SKOptLearner:
        """Return a new `~adaptive.SKOptLearner` without the data."""
        return SKOptLearner(self.function, **self._kwargs)

    def tell(self, x, y, fit=True):
        if isinstance(x, collections.abc.Iterable):
            self.pending_points.discard(tuple(x))
            self.data[tuple(x)] = y
            super().tell(x, y, fit)
        else:
            self.pending_points.discard(x)
            self.data[x] = y
            super().tell([x], y, fit)

    def tell_pending(self, x):
        # 'skopt.Optimizer' takes care of points we
        # have not got results for.
        self.pending_points.add(tuple(x))

    def remove_unfinished(self):
        pass

    @cache_latest
    def loss(self, real=True):
        if not self.models:
            return np.inf
        else:
            model = self.models[-1]
            # Return the in-sample error (i.e. test the model
            # with the training data). This is not the best
            # estimator of loss, but it is the cheapest.
            return 1 - model.score(self.Xi, self.yi)

    def ask(self, n, tell_pending=True):
        if not tell_pending:
            raise NotImplementedError(
                "Asking points is an irreversible "
                "action, so use `ask(n, tell_pending=True`."
            )
        points = super().ask(n)
        # TODO: Choose a better estimate for the loss improvement.
        if self.space.n_dims > 1:
            return points, [self.loss() / n] * n
        else:
            return [p[0] for p in points], [self.loss() / n] * n

    @property
    def npoints(self):
        """Number of evaluated points."""
        return len(self.Xi)

    def plot(self, nsamples=200):
        hv = ensure_holoviews()
        if self.space.n_dims > 1:
            raise ValueError("Can only plot 1D functions")
        bounds = self.space.bounds[0]
        if not self.Xi:
            p = hv.Scatter([]) * hv.Curve([]) * hv.Area([])
        else:
            scatter = hv.Scatter(([p[0] for p in self.Xi], self.yi))
            if self.models:
                model = self.models[-1]
                xs = np.linspace(*bounds, nsamples)
                xsp = self.space.transform(xs.reshape(-1, 1).tolist())
                y_pred, sigma = model.predict(xsp, return_std=True)
                # Plot model prediction for function
                curve = hv.Curve((xs, y_pred)).opts(line_dash="dashed")
                # Plot 95% confidence interval as colored area around points
                area = hv.Area(
                    (xs, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma),
                    vdims=["y", "y2"],
                ).opts(alpha=0.5, line_alpha=0)

            else:
                area = hv.Area([])
                curve = hv.Curve([])
            p = scatter * curve * area

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (bounds[1] - bounds[0])
        plot_bounds = (bounds[0] - margin, bounds[1] + margin)

        return p.redim(x={"range": plot_bounds})

    def _get_data(self):
        return [x[0] for x in self.Xi], self.yi

    def _set_data(self, data):
        xs, ys = data
        self.tell_many(xs, ys)

    def to_dataframe(  # type: ignore[override]
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
        TODO

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If `pandas` is not installed.
        """
        raise NotImplementedError

    def load_dataframe(  # type: ignore[override]
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
        TODO
        """
        raise NotImplementedError
