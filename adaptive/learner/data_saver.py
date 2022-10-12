from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Any, Callable

from adaptive.learner.base_learner import BaseLearner
from adaptive.utils import copy_docstring_from

try:
    import pandas

    with_pandas = True

except ModuleNotFoundError:
    with_pandas = False


def _to_key(x):
    return tuple(x.values) if x.values.size > 1 else x.item()


class DataSaver:
    """Save extra data associated with the values that need to be learned.

    Parameters
    ----------
    learner : `~adaptive.BaseLearner` instance
        The learner that needs to be wrapped.
    arg_picker : function
        Function that returns the argument that needs to be learned.

    Example
    -------
    Imagine we have a function that returns a dictionary
    of the form: ``{'y': y, 'err_est': err_est}``.

    >>> from operator import itemgetter
    >>> _learner = Learner1D(f, bounds=(-1.0, 1.0))
    >>> learner = DataSaver(_learner, arg_picker=itemgetter('y'))
    """

    def __init__(self, learner: BaseLearner, arg_picker: Callable) -> None:
        self.learner = learner
        self.extra_data = OrderedDict()
        self.function = learner.function
        self.arg_picker = arg_picker

    def new(self) -> DataSaver:
        """Return a new `DataSaver` with the same `arg_picker` and `learner`."""
        return DataSaver(self.learner.new(), self.arg_picker)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.learner, attr)

    @copy_docstring_from(BaseLearner.tell)
    def tell(self, x: Any, result: Any) -> None:
        y = self.arg_picker(result)
        self.extra_data[x] = result
        self.learner.tell(x, y)

    @copy_docstring_from(BaseLearner.tell_pending)
    def tell_pending(self, x: Any) -> None:
        self.learner.tell_pending(x)

    def to_dataframe(
        self, extra_data_name: str = "extra_data", **kwargs: Any
    ) -> pandas.DataFrame:
        """Return the data as a concatenated `pandas.DataFrame` from child learners.

        Parameters
        ----------
        extra_data_name : str, optional
            The name of the column containing the extra data, by default "extra_data".
        **kwargs : dict
            Keyword arguments passed to the ``child_learner.to_dataframe(**kwargs)``.

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
        df = self.learner.to_dataframe(**kwargs)

        df[extra_data_name] = [
            self.extra_data[_to_key(x)] for _, x in df[df.attrs["inputs"]].iterrows()
        ]
        return df

    def load_dataframe(
        self,
        df: pandas.DataFrame,
        extra_data_name: str = "extra_data",
        input_names: tuple[str] = (),
        **kwargs,
    ) -> None:
        """Load the data from a `pandas.DataFrame` into the learner.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with the data to load.
        extra_data_name : str, optional
            The ``extra_data_name`` used in `to_dataframe`, by default "extra_data".
        input_names : tuple[str], optional
            The input names of the child learner. By default the input names are
            taken from ``df.attrs["inputs"]``, however, metadata is not preserved
            when saving/loading a DataFrame to/from a file. In that case, the input
            names can be passed explicitly. For example, for a 2D learner, this would
            be ``input_names=('x', 'y')``.
        **kwargs : dict
            Keyword arguments passed to each ``child_learner.load_dataframe(**kwargs)``.
        """
        self.learner.load_dataframe(df, **kwargs)
        keys = df.attrs.get("inputs", list(input_names))
        for _, x in df[keys + [extra_data_name]].iterrows():
            key = _to_key(x[:-1])
            self.extra_data[key] = x[-1]

    def _get_data(self) -> tuple[Any, OrderedDict]:
        return self.learner._get_data(), self.extra_data

    def _set_data(
        self,
        data: tuple[Any, OrderedDict],
    ) -> None:
        learner_data, self.extra_data = data
        self.learner._set_data(learner_data)

    def __getstate__(self) -> tuple[BaseLearner, Callable, OrderedDict]:
        return (
            self.learner,
            self.arg_picker,
            self.extra_data,
        )

    def __setstate__(self, state: tuple[BaseLearner, Callable, OrderedDict]) -> None:
        learner, arg_picker, extra_data = state
        self.__init__(learner, arg_picker)
        self.extra_data = extra_data

    @copy_docstring_from(BaseLearner.save)
    def save(self, fname, compress=True) -> None:
        # We copy this method because the 'DataSaver' is not a
        # subclass of the 'BaseLearner'.
        BaseLearner.save(self, fname, compress)

    @copy_docstring_from(BaseLearner.load)
    def load(self, fname, compress=True) -> None:
        # We copy this method because the 'DataSaver' is not a
        # subclass of the 'BaseLearner'.
        BaseLearner.load(self, fname, compress)


def _ds(learner_type, arg_picker, *args, **kwargs):
    args = args[2:]  # functools.partial passes the first 2 arguments in 'args'!
    return DataSaver(learner_type(*args, **kwargs), arg_picker)


def make_datasaver(learner_type, arg_picker):
    """Create a `DataSaver` of a `learner_type` that can be instantiated
    with the `learner_type`'s key-word arguments.

    Parameters
    ----------
    learner_type : `~adaptive.BaseLearner` type
        The learner type that needs to be wrapped.
    arg_picker : function
        Function that returns the argument that needs to be learned.

    Example
    -------
    Imagine we have a function that returns a dictionary
    of the form: ``{'y': y, 'err_est': err_est}``.

    >>> from operator import itemgetter
    >>> DataSaver = make_datasaver(Learner1D, arg_picker=itemgetter('y'))
    >>> learner = DataSaver(function=f, bounds=(-1.0, 1.0))

    Or when using `adaptive.BalancingLearner.from_product`:

    >>> learner_type = make_datasaver(adaptive.Learner1D,
    ...     arg_picker=itemgetter('y'))
    >>> learner = adaptive.BalancingLearner.from_product(
    ...     jacobi, learner_type, dict(bounds=(0, 1)), combos)
    """
    return functools.partial(_ds, learner_type, arg_picker)
