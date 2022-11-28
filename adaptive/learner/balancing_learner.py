from __future__ import annotations

import itertools
import numbers
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from functools import partial
from operator import itemgetter
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numpy as np

from adaptive.learner.base_learner import BaseLearner
from adaptive.notebook_integration import ensure_holoviews
from adaptive.utils import cache_latest, named_product, restore

try:
    from typing import Literal, TypeAlias
except ImportError:
    from typing_extensions import Literal, TypeAlias

try:
    import pandas

    with_pandas = True
except ModuleNotFoundError:
    with_pandas = False


def dispatch(child_functions: list[Callable], arg: Any) -> Any:
    index, x = arg
    return child_functions[index](x)


STRATEGY_TYPE: TypeAlias = Literal["loss_improvements", "loss", "npoints", "cycle"]

CDIMS_TYPE: TypeAlias = Union[
    Sequence[Dict[str, Any]],
    Tuple[Sequence[str], Sequence[Tuple[Any, ...]]],
    None,
]


class BalancingLearner(BaseLearner):
    r"""Choose the optimal points from a set of learners.

    Parameters
    ----------
    learners : sequence of `~adaptive.BaseLearner`\s
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

    Attributes
    ----------
    learners : list
        The sequence of `~adaptive.BaseLearner`\s.
    function : callable
        A function that calls the functions of the underlying learners.
        Its signature is ``function(learner_index, point)``.
    strategy : 'loss_improvements' (default), 'loss', 'npoints', or 'cycle'.
        The points that the `BalancingLearner` choses can be either based on:
        the best 'loss_improvements', the smallest total 'loss' of the
        child learners, the number of points per learner, using 'npoints',
        or by cycling through the learners one by one using 'cycle'.
        One can dynamically change the strategy while the simulation is
        running by changing the ``learner.strategy`` attribute.

    Notes
    -----
    This learner compares the `loss` calculated from the "child" learners.
    This requires that the 'loss' from different learners *can be meaningfully
    compared*. For the moment we enforce this restriction by requiring that
    all learners are the same type but (depending on the internals of the
    learner) it may be that the loss cannot be compared *even between learners
    of the same type*. In this case the `~adaptive.BalancingLearner` will
    behave in an undefined way. Change the `strategy` in that case.
    """

    def __init__(
        self,
        learners: list[BaseLearner],
        *,
        cdims: CDIMS_TYPE = None,
        strategy: STRATEGY_TYPE = "loss_improvements",
    ) -> None:
        self.learners = learners

        # Naively we would make 'function' a method, but this causes problems
        # when using executors from 'concurrent.futures' because we have to
        # pickle the whole learner.
        self.function = partial(dispatch, [l.function for l in self.learners])  # type: ignore

        self._ask_cache = {}
        self._loss = {}
        self._pending_loss = {}
        self._cdims_default = cdims

        if len({learner.__class__ for learner in self.learners}) > 1:
            raise TypeError(
                "A BalacingLearner can handle only one type" " of learners."
            )

        self.strategy: STRATEGY_TYPE = strategy

    def new(self) -> BalancingLearner:
        """Create a new `BalancingLearner` with the same parameters."""
        return BalancingLearner(
            [learner.new() for learner in self.learners],
            cdims=self._cdims_default,
            strategy=self.strategy,
        )

    @property
    def data(self) -> dict[tuple[int, Any], Any]:
        data = {}
        for i, l in enumerate(self.learners):
            data.update({(i, p): v for p, v in l.data.items()})
        return data

    @property
    def pending_points(self) -> set[tuple[int, Any]]:
        pending_points = set()
        for i, l in enumerate(self.learners):
            pending_points.update({(i, p) for p in l.pending_points})
        return pending_points

    @property
    def npoints(self) -> int:
        return sum(l.npoints for l in self.learners)

    @property
    def nsamples(self):
        if hasattr(self.learners[0], "nsamples"):
            return sum(l.nsamples for l in self.learners)
        else:
            raise AttributeError(
                f"{type(self.learners[0])} as no attribute called `nsamples`."
            )

    @property
    def strategy(self) -> STRATEGY_TYPE:
        """Can be either 'loss_improvements' (default), 'loss', 'npoints', or
        'cycle'. The points that the `BalancingLearner` choses can be either
        based on: the best 'loss_improvements', the smallest total 'loss' of
        the child learners, the number of points per learner, using 'npoints',
        or by going through all learners one by one using 'cycle'.
        One can dynamically change the strategy while the simulation is
        running by changing the ``learner.strategy`` attribute."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: STRATEGY_TYPE) -> None:
        self._strategy = strategy
        if strategy == "loss_improvements":
            self._ask_and_tell = self._ask_and_tell_based_on_loss_improvements
        elif strategy == "loss":
            self._ask_and_tell = self._ask_and_tell_based_on_loss
        elif strategy == "npoints":
            self._ask_and_tell = self._ask_and_tell_based_on_npoints
        elif strategy == "cycle":
            self._ask_and_tell = self._ask_and_tell_based_on_cycle
            self._cycle = itertools.cycle(range(len(self.learners)))
        else:
            raise ValueError(
                'Only strategy="loss_improvements", strategy="loss",'
                ' strategy="npoints", or strategy="cycle" is implemented.'
            )

    def _ask_and_tell_based_on_loss_improvements(
        self, n: int
    ) -> tuple[list[tuple[int, Any]], list[float]]:
        selected = []  # tuples ((learner_index, point), loss_improvement)
        total_points = [l.npoints + len(l.pending_points) for l in self.learners]
        for _ in range(n):
            to_select = []
            for index, learner in enumerate(self.learners):
                # Take the points from the cache
                if index not in self._ask_cache:
                    self._ask_cache[index] = learner.ask(n=1, tell_pending=False)
                points, loss_improvements = self._ask_cache[index]
                to_select.append(
                    ((index, points[0]), (loss_improvements[0], -total_points[index]))
                )

            # Choose the optimal improvement.
            (index, point), (loss_improvement, _) = max(to_select, key=itemgetter(1))
            total_points[index] += 1
            selected.append(((index, point), loss_improvement))
            self.tell_pending((index, point))

        points, loss_improvements = map(list, zip(*selected))
        return points, loss_improvements

    def _ask_and_tell_based_on_loss(
        self, n: int
    ) -> tuple[list[tuple[int, Any]], list[float]]:
        selected = []  # tuples ((learner_index, point), loss_improvement)
        total_points = [l.npoints + len(l.pending_points) for l in self.learners]
        for _ in range(n):
            losses = self._losses(real=False)
            index, _ = max(
                enumerate(zip(losses, (-n for n in total_points))), key=itemgetter(1)
            )
            total_points[index] += 1

            # Take the points from the cache
            if index not in self._ask_cache:
                self._ask_cache[index] = self.learners[index].ask(n=1)
            points, loss_improvements = self._ask_cache[index]

            selected.append(((index, points[0]), loss_improvements[0]))
            self.tell_pending((index, points[0]))

        points, loss_improvements = map(list, zip(*selected))
        return points, loss_improvements

    def _ask_and_tell_based_on_npoints(
        self, n: numbers.Integral
    ) -> tuple[list[tuple[numbers.Integral, Any]], list[float]]:
        selected = []  # tuples ((learner_index, point), loss_improvement)
        total_points = [l.npoints + len(l.pending_points) for l in self.learners]
        for _ in range(n):
            index = np.argmin(total_points)
            # Take the points from the cache
            if index not in self._ask_cache:
                self._ask_cache[index] = self.learners[index].ask(n=1)
            points, loss_improvements = self._ask_cache[index]
            total_points[index] += 1
            selected.append(((index, points[0]), loss_improvements[0]))
            self.tell_pending((index, points[0]))

        points, loss_improvements = map(list, zip(*selected))
        return points, loss_improvements

    def _ask_and_tell_based_on_cycle(
        self, n: int
    ) -> tuple[list[tuple[numbers.Integral, Any]], list[float]]:
        points, loss_improvements = [], []
        for _ in range(n):
            index = next(self._cycle)
            point, loss_improvement = self.learners[index].ask(n=1)
            points.append((index, point[0]))
            loss_improvements.append(loss_improvement[0])
            self.tell_pending((index, point[0]))

        return points, loss_improvements

    def ask(
        self, n: int, tell_pending: bool = True
    ) -> tuple[list[tuple[numbers.Integral, Any]], list[float]]:
        """Chose points for learners."""
        if n == 0:
            return [], []

        if not tell_pending:
            with restore(*self.learners):
                return self._ask_and_tell(n)
        else:
            return self._ask_and_tell(n)

    def tell(self, x: tuple[numbers.Integral, Any], y: Any) -> None:
        index, x = x
        self._ask_cache.pop(index, None)
        self._loss.pop(index, None)
        self._pending_loss.pop(index, None)
        self.learners[index].tell(x, y)

    def tell_pending(self, x: tuple[numbers.Integral, Any]) -> None:
        index, x = x
        self._ask_cache.pop(index, None)
        self._loss.pop(index, None)
        self.learners[index].tell_pending(x)

    def _losses(self, real: bool = True) -> list[float]:
        losses = []
        loss_dict = self._loss if real else self._pending_loss

        for index, learner in enumerate(self.learners):
            if index not in loss_dict:
                loss_dict[index] = learner.loss(real)
            losses.append(loss_dict[index])

        return losses

    @cache_latest
    def loss(self, real: bool = True) -> float:
        losses = self._losses(real)
        return max(losses)

    def plot(
        self,
        cdims: CDIMS_TYPE = None,
        plotter: Callable[[BaseLearner], Any] | None = None,
        dynamic: bool = True,
    ):
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
            holoviews object. By default ``learner.plot()`` will be called.
        dynamic : bool, default True
            Return a `holoviews.core.DynamicMap` if True, else a
            `holoviews.core.HoloMap`. The `~holoviews.core.DynamicMap` is
            rendered as the sliders change and can therefore not be exported
            to html. The `~holoviews.core.HoloMap` does not have this problem.

        Returns
        -------
        dm : `holoviews.core.DynamicMap` (default) or `holoviews.core.HoloMap`
             A `DynamicMap` ``(dynamic=True)`` or `HoloMap`
             ``(dynamic=False)`` with sliders that are defined by `cdims`.
        """
        hv = ensure_holoviews()
        cdims = cdims or self._cdims_default

        if cdims is None:
            cdims = [{"i": i} for i in range(len(self.learners))]
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
        dm.cache_size = 1

        if dynamic:
            # XXX: change when https://github.com/pyviz/holoviews/issues/3637
            # is fixed.
            return dm.map(lambda obj: obj.opts(framewise=True), hv.Element)
        else:
            # XXX: change when https://github.com/ioam/holoviews/issues/3085
            # is fixed.
            vals = {d.name: d.values for d in dm.dimensions() if d.values}
            return hv.HoloMap(dm.select(**vals))

    def remove_unfinished(self) -> None:
        """Remove uncomputed data from the learners."""
        for learner in self.learners:
            learner.remove_unfinished()

    @classmethod
    def from_product(
        cls,
        f,
        learner_type: BaseLearner,
        learner_kwargs: dict[str, Any],
        combos: dict[str, Sequence[Any]],
    ) -> BalancingLearner:
        """Create a `BalancingLearner` with learners of all combinations of
        named variables’ values. The `cdims` will be set correctly, so calling
        `learner.plot` will be a `holoviews.core.HoloMap` with the correct labels.

        Parameters
        ----------
        f : callable
            Function to learn, must take arguments provided in in `combos`.
        learner_type : `BaseLearner`
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
        as ``adaptive.utils.named_product(**combos)``.
        """
        learners = []
        arguments = named_product(**combos)
        for combo in arguments:
            learner = learner_type(function=partial(f, **combo), **learner_kwargs)
            learners.append(learner)
        return cls(learners, cdims=arguments)

    def to_dataframe(self, index_name: str = "learner_index", **kwargs):
        """Return the data as a concatenated `pandas.DataFrame` from child learners.

        Parameters
        ----------
        index_name : str, optional
            The name of the index column indicating the learner index,
            by default "learner_index".
        **kwargs : dict
            Keyword arguments passed to each ``child_learner.to_dataframe(**kwargs)``.

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
        dfs = []
        for i, learner in enumerate(self.learners):
            df = learner.to_dataframe(**kwargs)
            cols = list(df.columns)
            df[index_name] = i
            df = df[[index_name] + cols]
            dfs.append(df)
        df = pandas.concat(dfs, axis=0, ignore_index=True)
        return df

    def load_dataframe(
        self, df: pandas.DataFrame, index_name: str = "learner_index", **kwargs
    ):
        """Load the data from a `pandas.DataFrame` into the child learners.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with the data to load.
        index_name : str, optional
            The ``index_name`` used in `to_dataframe`, by default "learner_index".
        **kwargs : dict
            Keyword arguments passed to each ``child_learner.load_dataframe(**kwargs)``.
        """
        for i, gr in df.groupby(index_name):
            self.learners[i].load_dataframe(gr, **kwargs)

    def save(
        self,
        fname: Callable[[BaseLearner], str] | Sequence[str],
        compress: bool = True,
    ) -> None:
        """Save the data of the child learners into pickle files
        in a directory.

        Parameters
        ----------
        fname: callable or sequence of strings
            Given a learner, returns a filename into which to save the data.
            Or a list (or iterable) with filenames.
        compress : bool, default True
            Compress the data upon saving using `gzip`. When saving
            using compression, one must load it with compression too.

        Example
        -------
        >>> def combo_fname(learner):
        ...     val = learner.function.keywords  # because functools.partial
        ...     fname = '__'.join([f'{k}_{v}.pickle' for k, v in val.items()])
        ...     return 'data_folder/' + fname
        >>>
        >>> def f(x, a, b): return a * x**2 + b
        >>>
        >>> learners = [Learner1D(functools.partial(f, **combo), (-1, 1))
        ...             for combo in adaptive.utils.named_product(a=[1, 2], b=[1])]
        >>>
        >>> learner = BalancingLearner(learners)
        >>> # Run the learner
        >>> runner = adaptive.Runner(learner)
        >>> # Then save
        >>> learner.save(combo_fname)  # use 'load' in the same way
        """
        if isinstance(fname, Iterable):
            for l, _fname in zip(self.learners, fname):
                l.save(_fname, compress=compress)
        else:
            for l in self.learners:
                l.save(fname(l), compress=compress)

    def load(
        self,
        fname: Callable[[BaseLearner], str] | Sequence[str],
        compress: bool = True,
    ) -> None:
        """Load the data of the child learners from pickle files
        in a directory.

        Parameters
        ----------
        fname: callable or sequence of strings
            Given a learner, returns a filename from which to load the data.
            Or a list (or iterable) with filenames.
        compress : bool, default True
            If the data is compressed when saved, one must load it
            with compression too.

        Example
        -------
        See the example in the `BalancingLearner.save` doc-string.
        """
        if isinstance(fname, Iterable):
            for l, _fname in zip(self.learners, fname):
                l.load(_fname, compress=compress)
        else:
            for l in self.learners:
                l.load(fname(l), compress=compress)

    def _get_data(self) -> list[Any]:
        return [l._get_data() for l in self.learners]

    def _set_data(self, data: list[Any]):
        for l, _data in zip(self.learners, data):
            l._set_data(_data)

    def __getstate__(self) -> tuple[list[BaseLearner], CDIMS_TYPE, STRATEGY_TYPE]:
        return (
            self.learners,
            self._cdims_default,
            self.strategy,
        )

    def __setstate__(self, state: tuple[list[BaseLearner], CDIMS_TYPE, STRATEGY_TYPE]):
        learners, cdims, strategy = state
        self.__init__(learners, cdims=cdims, strategy=strategy)
