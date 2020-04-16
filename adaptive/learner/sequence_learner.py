from copy import copy

from sortedcontainers import SortedDict, SortedSet

from adaptive.learner.base_learner import BaseLearner


class _IndexToPoint:
    """Call function with index of sequence.

    The SequenceLearner's function receives a tuple ``(index, point)``
    but the original function only takes ``point``.

    This is the same as `lambda x: function(x[1])`, however, that is not
    pickable.
    """

    def __init__(self, function, sequence):
        self.function = function
        self.sequence = sequence

    def __call__(self, index, *args, **kwargs):
        return self.function(self.sequence[index], *args, **kwargs)

    def __getstate__(self):
        return self.function, self.sequence

    def __setstate__(self, state):
        self.__init__(*state)


class SequenceLearner(BaseLearner):
    r"""A learner that will learn a sequence. It simply returns
    the points in the provided sequence when asked.

    This is useful when your problem cannot be formulated in terms of
    another adaptive learner, but you still want to use Adaptive's
    routines to run, save, and plot.

    Parameters
    ----------
    function : callable
        The function to learn. Must take a single element of `sequence`.
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
        self.function = _IndexToPoint(function, sequence)
        self._to_do_indices = SortedSet({i for i, _ in enumerate(sequence)})
        self._ntotal = len(sequence)
        self.sequence = copy(sequence)
        self.data = SortedDict()
        self.pending_points = set()

    def ask(self, n, tell_pending=True):
        indices = []
        loss_improvements = []
        for index in self._to_do_indices:
            if len(indices) >= n:
                break
            indices.append(index)
            loss_improvements.append(1 / self._ntotal)

        if tell_pending:
            for index in indices:
                self.tell_pending(index)

        return indices, loss_improvements

    def _get_data(self):
        return self.data

    def _set_data(self, data):
        if data:
            indices, values = zip(*data.items())
            self.tell_many(indices, values)

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

    def tell(self, index, value):
        self.data[index] = value
        self.pending_points.discard(index)
        self._to_do_indices.discard(index)

    def tell_pending(self, index):
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
