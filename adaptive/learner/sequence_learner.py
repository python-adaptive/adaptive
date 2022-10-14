from __future__ import annotations

from copy import copy
from numbers import Integral as Int
from typing import Any, Tuple

import cloudpickle
from sortedcontainers import SortedDict, SortedSet

from adaptive.learner.base_learner import BaseLearner
from adaptive.utils import assign_defaults, partial_function_from_dataframe

try:
    import pandas

    with_pandas = True

except ModuleNotFoundError:
    with_pandas = False

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias


PointType: TypeAlias = Tuple[Int, Any]


class _IgnoreFirstArgument:
    """Remove the first argument from the call signature.

    The SequenceLearner's function receives a tuple ``(index, point)``
    but the original function only takes ``point``.

    This is the same as `lambda x: function(x[1])`, however, that is not
    pickable.
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, index_point: PointType, *args, **kwargs):
        index, point = index_point
        return self.function(point, *args, **kwargs)

    def __getstate__(self):
        return self.function

    def __setstate__(self, function):
        self.__init__(function)


class SequenceLearner(BaseLearner):
    r"""A learner that will learn a sequence. It simply returns
    the points in the provided sequence when asked.

    This is useful when your problem cannot be formulated in terms of
    another adaptive learner, but you still want to use Adaptive's
    routines to run, save, and plot.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single element `sequence`.
    sequence : sequence
        The sequence to learn.

    Attributes
    ----------
    data : dict
        The data as a mapping from "index of element in sequence" => value.

    Notes
    -----
    From primitive tests, the `~adaptive.SequenceLearner` appears to have a
    similar performance to `ipyparallel`\s ``load_balanced_view().map``. With
    the added benefit of having results in the local kernel already.
    """

    def __init__(self, function, sequence):
        self._original_function = function
        self.function = _IgnoreFirstArgument(function)
        # prefer range(len(...)) over enumerate to avoid slowdowns
        # when passing lazy sequences
        self._to_do_indices = SortedSet(range(len(sequence)))
        self._ntotal = len(sequence)
        self.sequence = copy(sequence)
        self.data = SortedDict()
        self.pending_points = set()

    def new(self) -> SequenceLearner:
        """Return a new `~adaptive.SequenceLearner` without the data."""
        return SequenceLearner(self._original_function, self.sequence)

    def ask(
        self, n: int, tell_pending: bool = True
    ) -> tuple[list[PointType], list[float]]:
        indices = []
        points = []
        loss_improvements = []
        for index in self._to_do_indices:
            if len(points) >= n:
                break
            point = self.sequence[index]
            indices.append(index)
            points.append((index, point))
            loss_improvements.append(1 / self._ntotal)

        if tell_pending:
            for i, p in zip(indices, points):
                self.tell_pending((i, p))

        return points, loss_improvements

    def loss(self, real: bool = True) -> float:
        if not (self._to_do_indices or self.pending_points):
            return 0.0
        else:
            npoints = self.npoints + (0 if real else len(self.pending_points))
            return (self._ntotal - npoints) / self._ntotal

    def remove_unfinished(self) -> None:
        for i in self.pending_points:
            self._to_do_indices.add(i)
        self.pending_points = set()

    def tell(self, point: PointType, value: Any) -> None:
        index, point = point
        self.data[index] = value
        self.pending_points.discard(index)
        self._to_do_indices.discard(index)

    def tell_pending(self, point: PointType) -> None:
        index, point = point
        self.pending_points.add(index)
        self._to_do_indices.discard(index)

    def done(self) -> bool:
        return not self._to_do_indices and not self.pending_points

    def result(self) -> list[Any]:
        """Get the function values in the same order as ``sequence``."""
        if not self.done():
            raise Exception("Learner is not yet complete.")
        return list(self.data.values())

    @property
    def npoints(self) -> int:
        return len(self.data)

    def to_dataframe(
        self,
        with_default_function_args: bool = True,
        function_prefix: str = "function.",
        index_name: str = "i",
        x_name: str = "x",
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
        index_name : str, optional
            Name of the index parameter, by default "i"
        x_name : str, optional
            Name of the input value, by default "x"
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
        indices, ys = zip(*self.data.items()) if self.data else ([], [])
        sequence = [self.sequence[i] for i in indices]
        df = pandas.DataFrame(indices, columns=[index_name])
        df[x_name] = sequence
        df[y_name] = ys
        df.attrs["inputs"] = [index_name]
        df.attrs["output"] = y_name
        if with_default_function_args:
            assign_defaults(self._original_function, df, function_prefix)
        return df

    def load_dataframe(
        self,
        df: pandas.DataFrame,
        with_default_function_args: bool = True,
        function_prefix: str = "function.",
        index_name: str = "i",
        x_name: str = "x",
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
        index_name : str, optional
            The ``index_name`` used in ``to_dataframe``, by default "i"
        x_name : str, optional
            The ``x_name`` used in ``to_dataframe``, by default "x"
        y_name : str, optional
            The ``y_name`` used in ``to_dataframe``, by default "y"
        """
        indices = df[index_name].values
        xs = df[x_name].values
        self.tell_many(zip(indices, xs), df[y_name].values)
        if with_default_function_args:
            self.function = partial_function_from_dataframe(
                self._original_function, df, function_prefix
            )

    def _get_data(self) -> dict[int, Any]:
        return self.data

    def _set_data(self, data: dict[int, Any]) -> None:
        if data:
            indices, values = zip(*data.items())
            # the points aren't used by tell, so we can safely pass None
            points = [(i, None) for i in indices]
            self.tell_many(points, values)

    def __getstate__(self):
        return (
            cloudpickle.dumps(self._original_function),
            self.sequence,
            self._get_data(),
        )

    def __setstate__(self, state):
        function, sequence, data = state
        function = cloudpickle.loads(function)
        self.__init__(function, sequence)
        self._set_data(data)
