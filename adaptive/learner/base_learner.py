# -*- coding: utf-8 -*-
import abc
import collections
from copy import deepcopy


class BaseLearner(metaclass=abc.ABCMeta):
    """Base class for algorithms for learning a function 'f: X → Y'.

    Attributes
    ----------
    function : callable: X → Y
        The function to learn.
    data : dict: X → Y
        'function' evaluated at certain points.
        The values can be 'None', which indicates that the point
        will be evaluated, but that we do not have the result yet.
    npoints : int, optional
        The number of evaluated points that have been added to the learner.
        Subclasses do not *have* to implement this attribute.

    Subclasses may define a 'plot' method that takes no parameters
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
    def ask(self, n, add_data=True):
        """Choose the next 'n' points to evaluate.

        Parameters
        ----------
        n : int
            The number of points to choose.
        add_data : bool, default: True
            If True, add the chosen points to this
            learner's 'data' with 'None' for the 'y'
            values. Set this to False if you do not
            want to modify the state of the learner.
        """
        pass

    def __getstate__(self):
        return deepcopy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = state
