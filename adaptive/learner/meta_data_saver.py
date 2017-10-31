# -*- coding: utf-8 -*-

class DataSaver:
    """Save meta data associated with the values that need to be learned.

    Parameters
    ----------
    learner : Learner object
        The learner that needs to be wrapped.
    arg_picker : function
        Function that returns the argument that needs to be learned.

    Example
    -------
    Imagine we have a function that returns a dictionary
    of the form: `{'y': y, 'err_est': err_est}`.

    >>> _learner = Learner1D(f, bounds=(-1.0, 1.0))
    >>> learner = DataSaver(_learner, arg_picker=operator.itemgetter('y'))
    """

    def __init__(self, learner, arg_picker):
        super().__init__()
        self.learner = learner
        self.meta_data = {}
        self.function = learner.function
        self.arg_picker = arg_picker

        # The methods a subclass of the BaseLearner needs to implement
        self.choose_points = self.learner.choose_points
        self.loss = self.learner.loss
        self.remove_unfinished = self.learner.remove_unfinished

    def add_point(self, x, y):
        result = self.arg_picker(y)
        self.meta_data[x] = y
        self.learner.add_point(x, result)
