#-------------------------------------------------------------------------------
# Filename:     learner1D.py
# Description:  Contains 'Learner1D' object, a learner for 1D data.
#               TODO:
#-------------------------------------------------------------------------------

import heapq
from math import sqrt
import itertools
import numpy as np

class Learner1D(object):
    """ Learns and predicts a 1D function.

    Description
    -----------
    Answers questions like:
    * "How much data do you need to get 2% accuracy?"
    * "What is the current status?"
    * "If I give you n data points, which ones would you like?"
    (initialise/request/promise/put/describe current state)

    """

    def __init__(self, xdata=None, ydata=None):
        """Initialize the learner.

        Parameters
        ----------
        data :
           Possibly empty list of float-like tuples, describing the initial
           data.
        """

        # Set internal variables

        # A dict storing the loss function for each interval x_n.
        self._losses = {}

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self._neighbors = {}
        # A dict {x_n: y_n} for quick checking of local
        # properties.
        self._ydata = {}

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [[np.inf, -np.inf], [np.inf, -np.inf]]
        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [0, 0]
        self._oldscale = [0, 0]

        self.unfinished = {}

        # Add initial data if provided
        if xdata is not None:
            self.add_data(xdata, ydata)

    def loss(self, x_left, x_right):
        """Calculate loss in the interval x_left, x_right.

        Currently returns the rescaled length of the interval. If one of the
        y-values is missing, returns 0 (so the intervals with missing data are
        never touched. This behavior should be improved later.
        """
        assert x_left < x_right and self._neighbors[x_left][1] == x_right
        try:
            y_right, y_left = self._ydata[x_right], self._ydata[x_left]
            return sqrt(((x_right - x_left) / self._scale[0])**2 +
                        ((y_right - y_left) / self._scale[1])**2)
        except TypeError:  # One of y-values is None.
            return 0

    def add_data(self, xvalues, yvalues):
        """Add data to the intervals.

        Parameters
        ----------
        xvalues : iterable of numbers
            Values of the x coordinate.
        yvalues : iterable of numbers and None
            Values of the y coordinate. `None` means that the value will be
            provided later.
        """
        try:
            for x, y in zip(xvalues, yvalues):
                self.add_point(x, y)
        except TypeError:
            self.add_point(xvalues, yvalues)

    def add_point(self, x, y):
        """Update the data."""
        self._ydata[x] = y

        # Update the neighbors.
        if x not in self._neighbors:  # The point is new
            xvals = sorted(self._neighbors)
            pos = np.searchsorted(xvals, x)  # This could be done for multiple vals at once
            self._neighbors[None] = [None, None]  # To reduce the number of condititons.

            x_lower = xvals[pos-1] if pos != 0 else None
            x_upper = xvals[pos] if pos != len(xvals) else None

            self._neighbors[x] = [x_lower, x_upper]
            self._neighbors[x_lower][1] = x
            self._neighbors[x_upper][0] = x
            del self._neighbors[None]

        # Update the scale.
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        if y is not None:
            self._bbox[1][0] = min(self._bbox[1][0], y)
            self._bbox[1][1] = max(self._bbox[1][1], y)
        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

        # Update the losses.
        x_lower, x_upper = self._neighbors[x]
        if x_lower is not None:
            self._losses[x_lower, x] = self.loss(x_lower, x)
        if x_upper is not None:
            self._losses[x, x_upper] = self.loss(x, x_upper)
        try:
            del self._losses[x_lower, x_upper]
        except KeyError:
            pass

        # If the scale has doubled, recompute all losses.
        if self._scale > self._oldscale * 2:
            self._losses = {key: self.loss(*key) for key in self._losses}
            self._oldscale = self._scale

    def choose_points(self, n=10, add_to_data=False):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        def points(x, n):
            return list(np.linspace(x[0], x[1], n, endpoint=False)[1:])

        # Calculate how many points belong to each interval.
        quals = [(-loss, x_range, 1) for (x_range, loss) in
                 self._losses.items()]
        heapq.heapify(quals)
        for point_number in range(n):
            quality, x, n = quals[0]
            heapq.heapreplace(quals, (quality * n / (n+1), x, n + 1))

        xs = sum((points(x, n) for quality, x, n in quals), [])

        # Add `None`s to data because then the same point will not be returned
        # upon a next request. This can be used for parallelization.
        if add_to_data:
            self.add_data(xs, itertools.repeat(None))

        return xs

    def get_status(self):
        """Report current status.

        So far just returns some internal variables [losses, intervals and
        data]
        """
        return self._losses, self._neighbors, self._ydata

    def get_results(self):
        """Work with distributed.client.Future objects."""
        for x, y in self.unfinished.items():
            if y.done():
                y = self.unfinished.pop(x).result()
                self.add_point(x, y)

    def add_futures(self, xs, ys):
        try:
            for x, y in zip(xs, ys):
                self.unfinished[x] = y
        except TypeError:
            self.unfinished[xs] = ys
