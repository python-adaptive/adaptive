# -*- coding: utf-8 -*-
from collections import defaultdict
from contextlib import suppress
from functools import partial
from operator import itemgetter

from .base_learner import BaseLearner
from ..notebook_integration import ensure_holoviews
from ..utils import restore, named_product


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

    def __init__(self, learners, *, cdims=None):
        self.learners = learners

        # Naively we would make 'function' a method, but this causes problems
        # when using executors from 'concurrent.futures' because we have to
        # pickle the whole learner.
        self.function = partial(dispatch, [l.function for l in self.learners])

        self._points = {}
        self._loss = {}
        self._cdims_default = cdims

        if len(set(learner.__class__ for learner in self.learners)) > 1:
            raise TypeError('A BalacingLearner can handle only one type'
                            'of learners.')

    def _ask_and_tell(self, n):
        points = []
        for _ in range(n):
            loss_improvements = []
            pairs = []
            for index, learner in enumerate(self.learners):
                if index not in self._points:
                    self._points[index] = learner.ask(
                        n=1, add_data=False)
                point, loss_improvement = self._points[index]
                loss_improvements.append(loss_improvement[0])
                pairs.append((index, point[0]))
            x, _ = max(zip(pairs, loss_improvements), key=itemgetter(1))
            points.append(x)
            self.tell(x, None)

        return points, None

    def ask(self, n, add_data=True):
        """Chose points for learners."""
        if not add_data:
            with restore(*self.learners):
                return self._ask_and_tell(n)
        else:
            return self._ask_and_tell(n)

    def tell(self, x, y):
        index, x = x
        self._points.pop(index, None)
        self._loss.pop(index, None)
        self.learners[index].tell(x, y)

    def loss(self, real=True):
        losses = []
        for index, learner in enumerate(self.learners):
            if index not in self._loss:
                self._loss[index] = learner.loss(real)
            loss = self._loss[index]
            losses.append(loss)
        return max(losses)

    def plot(self, cdims=None, plotter=None):
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
        Returns
        -------
        dm : holoviews.DynamicMap object
            A DynamicMap with sliders that are defined by 'cdims'.
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
        return dm.redim.values(**d)

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
