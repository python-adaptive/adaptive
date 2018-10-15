# -*- coding: utf-8 -*-
from collections import defaultdict
from contextlib import suppress
from functools import partial
from operator import itemgetter

import numpy as np

from .base_learner import BaseLearner
from ..notebook_integration import ensure_holoviews
from ..utils import cache_latest, named_product, restore


def dispatch(child_functions, arg):
    index, x = arg
    return child_functions[index](x)


class BalancingLearner(BaseLearner):
    """Choose the optimal points from a set of learners.

    Parameters
    ----------
    learners : sequence of BaseLearner
        The learners from which to choose. These must all have the same type.
    cdims : sequence of dicts, or (keys, iterable of values), optional
        Constant dimensions; the parameters that label the learners. Used
        in `plot`.
        Example inputs that all give identical results:
        - sequence of dicts:
            >>> cdims = [{'A': True, 'B': 0},
            ...          {'A': True, 'B': 1},
            ...          {'A': False, 'B': 0},
            ...          {'A': False, 'B': 1}]`
        - tuple with (keys, iterable of values):
            >>> cdims = (['A', 'B'], itertools.product([True, False], [0, 1]))
            >>> cdims = (['A', 'B'], [(True, 0), (True, 1),
            ...                       (False, 0), (False, 1)])
    strategy : 'loss_improvements' (default), 'loss', or 'npoints'
        The points that the 'BalancingLearner' choses can be either based on:
        the best 'loss_improvements', the smallest total 'loss' of the
        child learners, or the number of points per learner, using 'npoints'.
        One can dynamically change the strategy while the simulation is
        running by changing the 'learner.strategy' attribute.

    Notes
    -----
    This learner compares the 'loss' calculated from the "child" learners.
    This requires that the 'loss' from different learners *can be meaningfully
    compared*. For the moment we enforce this restriction by requiring that
    all learners are the same type but (depending on the internals of the
    learner) it may be that the loss cannot be compared *even between learners
    of the same type*. In this case the BalancingLearner will behave in an
    undefined way.
    """

    def __init__(self, learners, *, cdims=None, strategy='loss_improvements'):
        self.learners = learners

        # Naively we would make 'function' a method, but this causes problems
        # when using executors from 'concurrent.futures' because we have to
        # pickle the whole learner.
        self.function = partial(dispatch, [l.function for l in self.learners])

        self._points = {}
        self._loss = {}
        self._pending_loss = {}
        self._cdims_default = cdims

        if len(set(learner.__class__ for learner in self.learners)) > 1:
            raise TypeError('A BalacingLearner can handle only one type'
                            ' of learners.')

        self.strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy
        if strategy == 'loss_improvements':
            self._ask_and_tell = self._ask_and_tell_based_on_loss_improvements
        elif strategy == 'loss':
            self._ask_and_tell = self._ask_and_tell_based_on_loss
        elif strategy == 'npoints':
            self._ask_and_tell = self._ask_and_tell_based_on_npoints
        else:
            raise ValueError(
                'Only strategy="loss_improvements", strategy="loss", or'
                ' strategy="npoints" is implemented.')

    def _ask_and_tell_based_on_loss_improvements(self, n):
        points = []
        loss_improvements = []
        for _ in range(n):
            improvements_per_learner = []
            pairs = []
            for index, learner in enumerate(self.learners):
                if index not in self._points:
                    self._points[index] = learner.ask(
                        n=1, tell_pending=False)
                point, loss_improvement = self._points[index]
                improvements_per_learner.append(loss_improvement[0])
                pairs.append((index, point[0]))
            x, l = max(zip(pairs, improvements_per_learner),
                       key=itemgetter(1))
            points.append(x)
            loss_improvements.append(l)
            self.tell_pending(x)

        return points, loss_improvements

    def _ask_and_tell_based_on_loss(self, n):
        points = []
        loss_improvements = []
        for _ in range(n):
            losses = self.losses(real=False)
            max_ind = np.argmax(losses)
            xs, ls = self.learners[max_ind].ask(1)
            points.append((max_ind, xs[0]))
            loss_improvements.append(ls[0])
        return points, loss_improvements

    def _ask_and_tell_based_on_npoints(self, n):
        points = []
        loss_improvements = []
        npoints = [l.npoints + len(l.pending_points)
                   for l in self.learners]
        n_left = n
        while n_left > 0:
            i = np.argmin(npoints)
            xs, ls = self.learners[i].ask(1)
            npoints[i] += 1
            n_left -= 1
            points.append((i, xs[0]))
            loss_improvements.append(ls[0])
        return points, loss_improvements

    def ask(self, n, tell_pending=True):
        """Chose points for learners."""
        if not tell_pending:
            with restore(*self.learners):
                return self._ask_and_tell(n)
        else:
            return self._ask_and_tell(n)

    def tell(self, x, y):
        index, x = x
        self._points.pop(index, None)
        self._loss.pop(index, None)
        self._pending_loss.pop(index, None)
        self.learners[index].tell(x, y)

    def tell_pending(self, x):
        index, x = x
        self._points.pop(index, None)
        self._loss.pop(index, None)
        self.learners[index].tell_pending(x)

    def losses(self, real=True):
        losses = []
        loss_dict = self._loss if real else self._pending_loss

        for index, learner in enumerate(self.learners):
            if index not in loss_dict:
                loss_dict[index] = learner.loss(real)
            losses.append(loss_dict[index])

        return losses

    @cache_latest
    def loss(self, real=True):
        losses = self.losses(real)
        return max(losses)

    def plot(self, cdims=None, plotter=None, dynamic=True):
        """Returns a DynamicMap with sliders.

        Parameters
        ----------
        cdims : sequence of dicts, or (keys, iterable of values), optional
            Constant dimensions; the parameters that label the learners.
            Example inputs that all give identical results:
            - sequence of dicts:
                >>> cdims = [{'A': True, 'B': 0},
                ...          {'A': True, 'B': 1},
                ...          {'A': False, 'B': 0},
                ...          {'A': False, 'B': 1}]`
            - tuple with (keys, iterable of values):
                >>> cdims = (['A', 'B'], itertools.product([True, False], [0, 1]))
                >>> cdims = (['A', 'B'], [(True, 0), (True, 1),
                ...                       (False, 0), (False, 1)])
        plotter : callable, optional
            A function that takes the learner as a argument and returns a
            holoviews object. By default learner.plot() will be called.
        dynamic : bool, default True
            Return a holoviews.DynamicMap if True, else a holoviews.HoloMap.
            The DynamicMap is rendered as the sliders change and can therefore
            not be exported to html. The HoloMap does not have this problem.

        Returns
        -------
        dm : holoviews.DynamicMap object (default) or holoviews.HoloMap object
            A DynamicMap (dynamic=True) or HoloMap (dynamic=False) with
            sliders that are defined by 'cdims'.
        """
        hv = ensure_holoviews()
        cdims = cdims or self._cdims_default

        if cdims is None:
            cdims = [{'i': i} for i in range(len(self.learners))]
        elif not isinstance(cdims[0], dict):
            # Normalize the format
            keys, values_list = cdims
            cdims = [dict(zip(keys, values)) for values in values_list]

        mapping = {tuple(_cdims.values()): l for l, _cdims in zip(self.learners, cdims)}

        d = defaultdict(list)
        for _cdims in cdims:
            for k, v in _cdims.items():
                d[k].append(v)

        def plot_function(*args):
            with suppress(KeyError):
                learner = mapping[tuple(args)]
                return learner.plot() if plotter is None else plotter(learner)

        dm = hv.DynamicMap(plot_function, kdims=list(d.keys()))
        dm = dm.redim.values(**d)

        if dynamic:
            return dm
        else:
            # XXX: change when https://github.com/ioam/holoviews/issues/3085
            # is fixed.
            vals = {d.name: d.values for d in dm.dimensions() if d.values}
            return hv.HoloMap(dm.select(**vals))

    def remove_unfinished(self):
        """Remove uncomputed data from the learners."""
        for learner in self.learners:
            learner.remove_unfinished()

    @classmethod
    def from_product(cls, f, learner_type, learner_kwargs, combos):
        """Create a `BalancingLearner` with learners of all combinations of
        named variables’ values. The `cdims` will be set correctly, so calling
        `learner.plot` will be a `holoviews.HoloMap` with the correct labels.

        Parameters
        ----------
        f : callable
            Function to learn, must take arguments provided in in `combos`.
        learner_type : BaseLearner
            The learner that should wrap the function. For example `Learner1D`.
        learner_kwargs : dict
            Keyword argument for the `learner_type`. For example `dict(bounds=[0, 1])`.
        combos : dict (mapping individual fn arguments -> sequence of values)
            For all combinations of each argument a learner will be instantiated.

        Returns
        -------
        learner : `BalancingLearner`
            A `BalancingLearner` with learners of all combinations of `combos`

        Example
        -------
        >>> def f(x, n, alpha, beta):
        ...     return scipy.special.eval_jacobi(n, alpha, beta, x)

        >>> combos = {
        ...     'n': [1, 2, 4, 8, 16],
        ...     'alpha': np.linspace(0, 2, 3),
        ...     'beta': np.linspace(0, 1, 5),
        ... }

        >>> learner = BalancingLearner.from_product(
        ...     f, Learner1D, dict(bounds=(0, 1)), combos)

        Notes
        -----
        The order of the child learners inside `learner.learners` is the same
        as `adaptive.utils.named_product(**combos)`.
        """
        learners = []
        arguments = named_product(**combos)
        for combo in arguments:
            learner = learner_type(function=partial(f, **combo), **learner_kwargs)
            learners.append(learner)
        return cls(learners, cdims=arguments)
