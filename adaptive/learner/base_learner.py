import abc
from contextlib import suppress
from copy import deepcopy
from typing import Any, Callable, Dict

from adaptive.utils import _RequireAttrsABCMeta, load, save


def uses_nth_neighbors(n: int) -> Callable:
    """Decorator to specify how many neighboring intervals the loss function uses.

    Wraps loss functions to indicate that they expect intervals together
    with ``n`` nearest neighbors

    The loss function will then receive the data of the N nearest neighbors
    (``nth_neighbors``) aling with the data of the interval itself in a dict.
    The `~adaptive.Learner1D` will also make sure that the loss is updated
    whenever one of the ``nth_neighbors`` changes.

    Examples
    --------

    The next function is a part of the `curvature_loss_function` function.

    >>> @uses_nth_neighbors(1)
    ... def triangle_loss(xs, ys):
    ...    xs = [x for x in xs if x is not None]
    ...    ys = [y for y in ys if y is not None]
    ...
    ...    if len(xs) == 2: # we do not have enough points for a triangle
    ...        return xs[1] - xs[0]
    ...
    ...    N = len(xs) - 2 # number of constructed triangles
    ...    if isinstance(ys[0], Iterable):
    ...        pts = [(x, *y) for x, y in zip(xs, ys)]
    ...        vol = simplex_volume_in_embedding
    ...    else:
    ...        pts = [(x, y) for x, y in zip(xs, ys)]
    ...        vol = volume
    ...    return sum(vol(pts[i:i+3]) for i in range(N)) / N

    Or you may define a loss that favours the (local) minima of a function,
    assuming that you know your function will have a single float as output.

    >>> @uses_nth_neighbors(1)
    ... def local_minima_resolving_loss(xs, ys):
    ...     dx = xs[2] - xs[1] # the width of the interval of interest
    ...
    ...     if not ((ys[0] is not None and ys[0] > ys[1])
    ...         or (ys[3] is not None and ys[3] > ys[2])):
    ...         return loss * 100
    ...
    ...     return loss
    """

    def _wrapped(loss_per_interval):
        loss_per_interval.nth_neighbors = n
        return loss_per_interval

    return _wrapped


class BaseLearner(metaclass=_RequireAttrsABCMeta):
    """Base class for algorithms for learning a function 'f: X → Y'.

    Attributes
    ----------
    function : callable: X → Y
        The function to learn. A subclass of BaseLearner might modify
        the user's supplied function.
    data : dict: X → Y
        `function` evaluated at certain points.
    pending_points : set
        Points that have been requested but have not been evaluated yet.
    npoints : int
        The number of evaluated points that have been added to the learner.

    Notes
    -----
    Subclasses may define a ``plot`` method that takes no parameters
    and returns a holoviews plot.
    """

    data: dict
    npoints: int
    pending_points: set

    def tell(self, x: Any, y) -> None:
        """Tell the learner about a single value.

        Parameters
        ----------
        x : A value from the function domain
        y : A value from the function image
        """
        self.tell_many([x], [y])

    def tell_many(self, xs: Any, ys: Any) -> None:
        """Tell the learner about some values.

        Parameters
        ----------
        xs : Iterable of values from the function domain
        ys : Iterable of values from the function image
        """
        for x, y in zip(xs, ys):
            self.tell(x, y)

    @abc.abstractmethod
    def tell_pending(self, x):
        """Tell the learner that 'x' has been requested such
        that it's not suggested again."""
        pass

    @abc.abstractmethod
    def remove_unfinished(self):
        """Remove uncomputed data from the learner."""
        pass

    @abc.abstractmethod
    def loss(self, real=True):
        """Return the loss for the current state of the learner.

        Parameters
        ----------
        real : bool, default: True
            If False, return the "expected" loss, i.e. the
            loss including the as-yet unevaluated points
            (possibly by interpolation).
        """

    @abc.abstractmethod
    def ask(self, n, tell_pending=True):
        """Choose the next 'n' points to evaluate.

        Parameters
        ----------
        n : int
            The number of points to choose.
        tell_pending : bool, default: True
            If True, add the chosen points to this learner's
            `pending_points`. Set this to False if you do not
            want to modify the state of the learner.
        """
        pass

    @abc.abstractmethod
    def _get_data(self):
        pass

    @abc.abstractmethod
    def _set_data(self):
        pass

    def copy_from(self, other):
        """Copy over the data from another learner.

        Parameters
        ----------
        other : BaseLearner object
            The learner from which the data is copied.
        """
        self._set_data(other._get_data())

    def save(self, fname: str, compress: bool = True) -> None:
        """Save the data of the learner into a pickle file.

        Parameters
        ----------
        fname : str
            The filename into which to save the learner's data.
        compress : bool, default True
            Compress the data upon saving using 'gzip'. When saving
            using compression, one must load it with compression too.
        """
        data = self._get_data()
        save(fname, data, compress)

    def load(self, fname: str, compress: bool = True) -> None:
        """Load the data of a learner from a pickle file.

        Parameters
        ----------
        fname : str
            The filename from which to load the learner's data.
        compress : bool, default True
            If the data is compressed when saved, one must load it
            with compression too.
        """
        with suppress(FileNotFoundError, EOFError):
            data = load(fname, compress)
            self._set_data(data)

    def __getstate__(self) -> Dict[str, Any]:
        return deepcopy(self.__dict__)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
