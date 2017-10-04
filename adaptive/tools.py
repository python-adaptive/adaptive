import functools

class cached_property_if_true:
    """A property that is computed each time, untill the result is true,
    then it replaces itself with an ordinary attribute.
    Deleting the attribute resets the property."""
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __get__(self, obj, cls):
        if obj is None: return self
        value = self.func(obj)
        if value:
            obj.__dict__[self.func.__name__] = value
        return value