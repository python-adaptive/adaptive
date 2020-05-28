from copy import copy

import cloudpickle
from sortedcontainers import SortedDict, SortedSet

from adaptive.learner.base_learner import BaseLearner


class _IgnoreFirstArgument:
    """Remove the first argument from the call signature.

    The SequenceLearner's function receives a tuple ``(index, point)``
    but the original function only takes ``point``.

    This is the same as `lambda x: function(x[1])`, however, that is not
    pickable.
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, index_point, *args, **kwargs):
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
        self._to_do_indices = SortedSet({i for i, _ in enumerate(sequence)})
        self._ntotal = len(sequence)
        self.sequence = copy(sequence)
        self.data = SortedDict()
        self.pending_points = set()

    def ask(self, n, tell_pending=True):
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

    def loss(self, real=True):
        if not (self._to_do_indices or self.pending_points):
            return 0
        else:
            npoints = self.npoints + (0 if real else len(self.pending_points))
            return (self._ntotal - npoints) / self._ntotal

    def remove_unfinished(self):
        for i in self.pending_points:
            self._to_do_indices.add(i)
        self.pending_points = set()

    def tell(self, point, value):
        index, point = point
        self.data[index] = value
        self.pending_points.discard(index)
        self._to_do_indices.discard(index)

    def tell_pending(self, point):
        index, point = point
        self.pending_points.add(index)
        self._to_do_indices.discard(index)

    def done(self):
        return not self._to_do_indices and not self.pending_points

    def result(self):
        """Get the function values in the same order as ``sequence``."""
        if not self.done():
            raise Exception("Learner is not yet complete.")
        return list(self.data.values())

    @property
    def npoints(self):
        return len(self.data)

    def _get_data(self):
        return self.data

    def _set_data(self, data):
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
