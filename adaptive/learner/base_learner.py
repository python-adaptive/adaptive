# -*- coding: utf-8 -*-
import abc
from contextlib import suppress
from copy import deepcopy

from ..utils import save, load


class BaseLearner(metaclass=abc.ABCMeta):
    """Base class for algorithms for learning a function 'f: X → Y'.

    Attributes
    ----------
    function : callable: X → Y
        The function to learn.
    data : dict: X → Y
        `function` evaluated at certain points.
        The values can be 'None', which indicates that the point
        will be evaluated, but that we do not have the result yet.
    npoints : int, optional
        The number of evaluated points that have been added to the learner.
        Subclasses do not *have* to implement this attribute.

    Notes
    -----
    Subclasses may define a ``plot`` method that takes no parameters
    and returns a holoviews plot.
    """

    def tell(self, x, y):
        """Tell the learner about a single value.

        Parameters
        ----------
        x : A value from the function domain
        y : A value from the function image
        """
        self.tell_many([x], [y])

    def tell_many(self, xs, ys):
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

    def save(self, fname=None, compress=True):
        """Save the data of the learner into a pickle file.

        Parameters
        ----------
        fname : str, optional
            The filename of the learner's pickle data file. If None use
            the 'fname' attribute, like 'learner.fname = "example.p".
        compress : bool, default True
            Compress the data upon saving using 'gzip'. When saving
            using compression, one must load it with compression too.

        Notes
        -----
        There are __two ways__ of naming the files:
        1. Using the 'fname' argument in 'learner.save(fname='example.p')
        2. Setting the 'fname' attribute, like
           'learner.fname = "data/example.p"' and then 'learner.save()'.
        """
        fname = fname or self.fname
        data = self._get_data()
        save(fname, data, compress)

    def load(self, fname=None, compress=True):
        """Load the data of a learner from a pickle file.

        Parameters
        ----------
        fname : str, optional
            The filename of the saved learner's pickled data file.
            If None use the 'fname' attribute, like
            'learner.fname = "example.p".
        compress : bool, default True
            If the data is compressed when saved, one must load it
            with compression too.

        Notes
        -----
        See the notes in the 'BaseLearner.save' doc-string.
        """
        fname = fname or self.fname
        with suppress(FileNotFoundError, EOFError):
            data = load(fname, compress)
            self._set_data(data)

    def __getstate__(self):
        return deepcopy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def fname(self):
        # This is a property because then it will be availible in the DataSaver
        try:
            return self._fname
        except AttributeError:
            raise AttributeError("Set 'learner.fname' or use the 'fname'"
                " argument when using 'learner.save' or 'learner.load'.")

    @fname.setter
    def fname(self, fname):
        self._fname = fname
