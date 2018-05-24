# -*- coding: utf-8 -*-
import abc
import collections
from copy import deepcopy
from ..notebook_integration import ensure_holoviews


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

    def tell(self, xvalues, yvalues):
        """Add data to the learner.

        Parameters
        ----------
        xvalues : value from the function domain, or iterable of such
            Values from the domain of the learned function.
        yvalues : value from the function image, or iterable of such
            Values from the range of the learned function, or None.
            If 'None', then it indicates that the value has not yet
            been computed.
        """
        if all(isinstance(i, collections.Iterable) for i in [xvalues, yvalues]):
            for x, y in zip(xvalues, yvalues):
                self._tell(x, y)
        else:
            self._tell(xvalues, yvalues)

    @abc.abstractmethod
    def _tell(self, x, y):
        """Add a single datapoint to the learner."""
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

    def matplotlibShow(self, *args, **kwargs):
        import holoviews.plotting.mpl
        hv = ensure_holoviews()
        p = self.plot(*args, **kwargs)
        renderer = hv.Store.renderers['matplotlib']
        renderer.show(p)


    def __getstate__(self):
        return deepcopy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = state
