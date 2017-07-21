# ------------------------------------------------------------------------------
# Filename:     learner1D.py
# Description:  Contains 'Learner1D' object, a learner for 1D data.
# ------------------------------------------------------------------------------

import heapq
from math import sqrt
import itertools
import multiprocessing

import numpy as np
import tornado


def add_arg(func):
    """Make func return (arg, func(arg))."""
    def wrapper(*args):
        return (args[0], func(*args))
    return wrapper


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
        self.losses = {}

        # A dict {x_n: [x_{n-1}, x_{n+1}]} for quick checking of local
        # properties.
        self.neighbors = {}
        # A dict {x_n: y_n} for quick checking of local
        # properties.
        self.data = {}

        # Bounding box [[minx, maxx], [miny, maxy]].
        self._bbox = [[np.inf, -np.inf], [np.inf, -np.inf]]
        # Data scale (maxx - minx), (maxy - miny)
        self._scale = [0, 0]
        self._oldscale = [0, 0]

        # Add initial data if provided
        if xdata is not None:
            self.add_data(xdata, ydata)

    def interval_loss(self, x_left, x_right):
        """Calculate loss in the interval x_left, x_right.

        Currently returns the rescaled length of the interval. If one of the
        y-values is missing, returns 0 (so the intervals with missing data are
        never touched. This behavior should be improved later.
        """
        y_right, y_left = self.interp_data[x_right], self.interp_data[x_left]
        return sqrt(((x_right - x_left) / self._scale[0])**2 +
                    ((y_right - y_left) / self._scale[1])**2)

    def loss(self):
        if len(self.losses) == 0:
            return float('inf')
        else:
            return max(self.losses.values())

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
        self.data[x] = y

        # Update the scale.
        self._bbox[0][0] = min(self._bbox[0][0], x)
        self._bbox[0][1] = max(self._bbox[0][1], x)
        self.x_range = self._bbox[0][1] - self._bbox[0][0]
        if y is not None:
            self._bbox[1][0] = min(self._bbox[1][0], y)
            self._bbox[1][1] = max(self._bbox[1][1], y)
        self._scale = [self._bbox[0][1] - self._bbox[0][0],
                       self._bbox[1][1] - self._bbox[1][0]]

    def choose_points(self, n=10):
        """Return n points that are expected to maximally reduce the loss."""
        # Find out how to divide the n points over the intervals
        # by finding  positive integer n_i that minimize max(L_i / n_i) subject
        # to a constraint that sum(n_i) = n + N, with N the total number of
        # intervals.

        # Return equally spaced points within each interval to which points
        # will be added.
        self.interpolate()  # Apply new interpolation step if new results

        def points(x, n):
            return list(np.linspace(x[0], x[1], n, endpoint=False)[1:])

        # Calculate how many points belong to each interval.
        quals = [(-loss, x_range, 1) for (x_range, loss) in
                 self.losses.items()]
        heapq.heapify(quals)

        for point_number in range(n):
            quality, x, n = quals[0]
            heapq.heapreplace(quals, (quality * n / (n + 1), x, n + 1))

        xs = sum((points(x, n) for quality, x, n in quals), [])

        # Add `None`s to data because then the same point will not be returned
        # upon a next request. This can be used for parallelization.
        self.add_data(xs, itertools.repeat(None))

        return xs

    def get_largest_interval(self):
        xs = sorted(x for x, y in self.data.items() if y is not None)
        if len(xs) < 2:
            return np.inf
        else:
            return np.diff(xs).max()

    def interpolate(self):
        xdata = []
        ydata = []
        xdata_unfinished = []
        self.interp_data = {}
        for x in sorted(self.data):
            y = self.data[x]
            if y is None:
                xdata_unfinished.append(x)
            else:
                xdata.append(x)
                ydata.append(y)
                self.interp_data[x] = y

        if len(ydata) == 0:
            ydata_unfinished = (0, ) * len(xdata_unfinished)
        else:
            ydata_unfinished = np.interp(xdata_unfinished, xdata, ydata)

        for x, y in zip(xdata_unfinished, ydata_unfinished):
            self.interp_data[x] = y

        self.neighbors = {}
        xdata_sorted = sorted(self.interp_data)
        for i, x in enumerate(xdata_sorted):
            if i == 0:
                self.neighbors[x] = [None, xdata_sorted[1]]
            elif i == len(xdata_sorted) - 1:
                self.neighbors[x] = [xdata_sorted[i-1], None]
            else:
                self.neighbors[x] = [xdata_sorted[i-1], xdata_sorted[i+1]]

        self.losses = {}
        for x, (x_left, x_right) in self.neighbors.items():
            if x_left is not None:
                self.losses[(x_left, x)] = self.interval_loss(x_left, x)
            if x_right is not None:
                self.losses[x, x_right] = self.interval_loss(x, x_right)
            try:
                del self.losses[x_left, x_right]
            except KeyError:
                pass


# We can't use API that is specific to any particular asynchronous
# framework, so we have to roll our own utility functions.

async def any_complete(futures):
    total = tornado.concurrent.Future()
    for f in futures:
        f.add_done_callback(total.set_result)
    await total
    return [f for f in futures if f.done()]


async def run(f, executor, learner, goal, ncores=multiprocessing.cpu_count()):
    xs = dict()
    done = [None] * ncores

    while not goal(learner):
        # Launch tasks to replace the ones that completed
        # on the last iteration.
        for x in learner.choose_points(len(done)):
            xs[executor.submit(f, x)] = x

        # Collect and results and add them to the learner
        futures = list(xs.keys())
        await any_complete(futures)
        done = [fut for fut in futures if fut.done()]
        for fut in done:
            x = xs.pop(fut)
            # Need to explicitly await the future (even though we know the
            # result is there) to be compatible with Dask, who's futures'
            # 'result' method return a future themselves.
            y = await fut
            learner.add_point(x, y)

    # cancel any outstanding tasks
    for fut in xs.keys():
        fut.cancel()
    # XXX: we should introduce an API for removing data points, and remove all
    #      the data points with a 'None' value from the learner, or add a
    #      method to simply remove all "unfinished" points from the learner.
