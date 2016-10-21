'''@package nnetgraph
contains the functionality to create neural network graphs and train/test it
'''
#turn off the too few public methods complaint for the interface.
#pylint: disable=R0903

from abc import ABCMeta, abstractmethod

class NnetGraph(object):
    """
    Abstract class difining an interface to be implementd by neural net graphs.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError("Abstract method")

    def __init__(self, output_dim):
        """ Every net graph should have the output dimension
            attribute set.
        """
        self.output_dim = output_dim
