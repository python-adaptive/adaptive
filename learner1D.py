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

        # A dict with {x_n: concurrent.futures}
        self.unfinished = {}

        # Add initial data if provided
        if xdata is not None:
            self.add_data(xdata, ydata)


    def loss(self, x_left, x_right, interpolate=False):
        """Calculate loss in the interval x_left, x_right.

        Currently returns the rescaled length of the interval. If one of the
        y-values is missing, returns 0 (so the intervals with missing data are
        never touched. This behavior should be improved later.
        """
        if interpolate:
            ydata = self.interp_ydata
            assert ydata.keys() == self._ydata.keys()
        else:
            ydata = self._ydata

        assert x_left < x_right and self._neighbors[x_left][1] == x_right

        try:
            y_right, y_left = ydata[x_right], ydata[x_left]
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

            x_left = xvals[pos-1] if pos != 0 else None
            x_right = xvals[pos] if pos != len(xvals) else None

            self._neighbors[x] = [x_left, x_right]
            self._neighbors[x_left][1] = x
            self._neighbors[x_right][0] = x
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
        x_left, x_right = self._neighbors[x]
        if x_left is not None:
            self._losses[x_left, x] = self.loss(x_left, x)
        if x_right is not None:
            self._losses[x, x_right] = self.loss(x, x_right)
        try:
            del self._losses[x_left, x_right]
        except KeyError:
            pass

        # If the scale has doubled, recompute all losses.
        if self._scale > self._oldscale * 2:
            self._losses = {key: self.loss(*key) for key in self._losses}
            self._oldscale = self._scale

    def choose_points(self, n=10, add_to_data=False, interpolate=False):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        if interpolate:
            self.interpolate()
            losses = self.interp_losses.items()
        else:
            losses = self._losses.items()

        def points(x, n):
            return list(np.linspace(x[0], x[1], n, endpoint=False)[1:])

        # Calculate how many points belong to each interval.
        quals = [(-loss, x_range, 1) for (x_range, loss) in
                 losses]
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
        done = [(x, y.result()) for x, y in self.unfinished.items() if y.done()]
        for x, y in done:
            self.unfinished.pop(x)
            self.add_point(x, y)

    def add_futures(self, xs, ys):
        """Add concurrent.futures to the self.unfinished dict."""
        try:
            for x, y in zip(xs, ys):
                self.unfinished[x] = y
        except TypeError:
            self.unfinished[xs] = ys


    def interpolate(self):
        """Estimates the approximate positions of unknown y-values by
        interpolating and assuming the unknown point lies on a line between
        its nearest known neighbors.

        Upon running this function it adds:
        self.interp_ydata
        self.interp_losses
        self.real_neighbors
        """
        ydata = sorted([x for x, y in self._ydata.items() if y is not None])

        self.real_neighbors = {}
        for i, y in enumerate(ydata):
            if i == 0:
                self.real_neighbors[y] = [None, ydata[1]]
            elif i == len(ydata) - 1:
                self.real_neighbors[y] = [ydata[i-1], None]
            else:
                self.real_neighbors[y] = [ydata[i-1], ydata[i+1]]

        ydata_unfinished = [x for x, y in self._ydata.items() if y is None]
        indices = np.searchsorted(ydata, ydata_unfinished)

        for i, y in zip(indices, ydata_unfinished):
            x_left, x_right = self.real_neighbors[ydata[i]]
            self.real_neighbors[y] = [x_left, ydata[i]]

        self.interp_ydata = {}
        for x, (x_left, x_right) in self.real_neighbors.items():
            y = self._ydata[x]
            if y is None:
                y_left = self._ydata[x_left]
                y_right = self._ydata[x_right]
                y = np.interp(x, [x_left, x_right], [y_left, y_right])
            self.interp_ydata[x] = y

        self.interp_losses = {}
        for x, (x_left, x_right) in self.real_neighbors.items():
            if x_left is not None:
                self.interp_losses[(x_left, x)] = self.loss(
                    x_left, x, interpolate=True)
            if x_right is not None:
                self.interp_losses[x, x_right] = self.loss(
                    x, x_right, interpolate=True)
            try:
                del self.interp_losses[x_left, x_right]
            except KeyError:
                pass
