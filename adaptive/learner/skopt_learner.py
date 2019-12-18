import collections
from typing import Callable, List, Tuple, Union

import numpy as np
from skopt import Optimizer

from adaptive.learner.base_learner import BaseLearner
from adaptive.notebook_integration import ensure_holoviews
from adaptive.utils import cache_latest


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

    def __init__(self, function: Callable, **kwargs) -> None:
        self.function = function
        self.pending_points = set()
        self.data = collections.OrderedDict()
        super().__init__(**kwargs)

    def tell(self, x: Union[float, List[float]], y: float, fit: bool = True) -> None:
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
    def loss(self, real: bool = True) -> float:
        if not self.models:
            return np.inf
        else:
            model = self.models[-1]
            # Return the in-sample error (i.e. test the model
            # with the training data). This is not the best
            # estimator of loss, but it is the cheapest.
            return 1 - model.score(self.Xi, self.yi)

    def ask(
        self, n: int, tell_pending: bool = True
    ) -> Union[
        Tuple[List[float], List[float]],
        Tuple[List[List[float]], List[float]],  # XXX: this indicates a bug!
    ]:
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
    def npoints(self) -> int:
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
                curve = hv.Curve((xs, y_pred)).opts(style=dict(line_dash="dashed"))
                # Plot 95% confidence interval as colored area around points
                area = hv.Area(
                    (xs, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma),
                    vdims=["y", "y2"],
                ).opts(style=dict(alpha=0.5, line_alpha=0))

            else:
                area = hv.Area([])
                curve = hv.Curve([])
            p = scatter * curve * area

        # Plot with 5% empty margins such that the boundary points are visible
        margin = 0.05 * (bounds[1] - bounds[0])
        plot_bounds = (bounds[0] - margin, bounds[1] + margin)

        return p.redim(x=dict(range=plot_bounds))

    def _get_data(self):
        return [x[0] for x in self.Xi], self.yi

    def _set_data(self, data):
        xs, ys = data
        self.tell_many(xs, ys)
