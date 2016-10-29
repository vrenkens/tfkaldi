'''@file classifier.py
The abstract class for a neural network classifier'''

from abc import ABCMeta, abstractmethod

class Classifier(object):
    '''This an abstract class defining a neural net classifier'''
    __metaclass__ = ABCMeta

    def __init__(self, output_dim):
        '''classifier constructor'''

        self.output_dim = output_dim

    @abstractmethod
    def __call__(self, inputs, seq_length, is_training=False, reuse=False,
                 scope=None):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a list containing
                a [batch_size, input_dim] tensor for each time step
            seq_length: The sequence lengths of the input utterances
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the name scope

        Returns:
            A triple containing:
                - output logits
                - the output logits sequence lengths as a vector
                - a saver object
                - a dictionary of control operations (may be empty)
        '''

        raise NotImplementedError("Abstract method")
