from copy import copy
import sys

from adaptive.learner.base_learner import BaseLearner

inf = sys.float_info.max


def ensure_hashable(x):
    try:
        hash(x)
        return x
    except TypeError:
        return tuple(x)


class SequenceLearner(BaseLearner):
    def __init__(self, function, sequence):
        self.function = function
        self._to_do_seq = {ensure_hashable(x) for x in sequence}
        self._npoints = len(sequence)
        self.sequence = copy(sequence)
        self.data = {}
        self.pending_points = set()

    def ask(self, n, tell_pending=True):
        points = []
        loss_improvements = []
        i = 0
        for point in self._to_do_seq:
            if i > n:
                break
            points.append(point)
            loss_improvements.append(inf / self._npoints)
            i += 1

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def _get_data(self):
        return self.data

    def _set_data(self, data):
        if data:
            self.tell_many(*zip(*data.items()))

    def loss(self, real=True):
        if not (self._to_do_seq or self.pending_points):
            return 0
        else:
            npoints = self.npoints + (0 if real else len(self.pending_points))
            return inf / npoints

    def remove_unfinished(self):
        for p in self.pending_points:
            self._to_do_seq.add(p)
        self.pending_points = set()

    def tell(self, point, value):
        self.data[point] = value
        self.pending_points.discard(point)
        self._to_do_seq.discard(point)

    def tell_pending(self, point):
        self.pending_points.add(point)
        self._to_do_seq.discard(point)

    def done(self):
        return not self._to_do_seq and not self.pending_points

    def result(self):
        """Get back the data in the same order as ``sequence``."""
        if not self.done():
            raise Exception("Learner is not yet complete.")
        return [self.data[ensure_hashable(x)] for x in self.sequence]

    @property
    def npoints(self):
        return len(self.data)
