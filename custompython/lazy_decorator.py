#stolen from https://danijar.com/structuring-your-tensorflow-models/
import functools


def lazy_property(function):
    ''' This decorator makes shure that
        the operations defined in the function
        are only added to the graph, when
        the function is called for the first time.'''
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

#class testProp:
#    def __init__(self, const):
#        self.const = const
#        self.plustwo
#
#    @lazy_property
#    def plustwo(self):
#        return self.const + 2
#
