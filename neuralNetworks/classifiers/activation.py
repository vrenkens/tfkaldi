'''@file activation.py
Activation functions for neural networks'''

from abc import ABCMeta, abstractmethod

import tensorflow as tf

class Activation(object):
    '''a class for activation functions'''
    __metaclass__ = ABCMeta

    def __init__(self, activation=None):
        '''
        Activation constructor
        Args:
            activation: the activation function being wrapped,
                if None, no activation will be wrapped
        '''

        self.activation = activation

    def __call__(self, inputs, is_training=False, reuse=False):
        '''
        apply the activation function
        Args:
            inputs: the inputs to the activation function
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
        Returns:
            the output to the activation function
        '''

        if self.activation is not None:
            #apply the wrapped activation
            activations = self.activation(inputs, is_training, reuse)
        else:
            activations = inputs

        #add own computation
        activation = self._apply_func(activations, is_training, reuse)

        return activation

    @abstractmethod
    def _apply_func(self, activations, is_training, reuse):
        '''
        apply own functionality
        Args:
            activations: the ioutputs to the wrapped activation function
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
        Returns:
            the output to the activation function
        '''

        raise NotImplementedError("Abstract method")

class TfActivation(Activation):
    '''a wrapper for an activation function that will add a tf activation
        function (e.g. relu, sigmoid, ...)'''

    def __init__(self, activation, tfActivation):
        '''
        the Tf_wrapper constructor
        Args:
            activation: the activation function being wrapped
            tfActivation: the tensorflow activation function that is wrapping
        '''

        super(TfActivation, self).__init__(activation)
        self.tf_activation = tfActivation

    def _apply_func(self, activations, is_training, reuse):
        '''
        apply own functionality
        Args:
            activations: the ioutputs to the wrapped activation function
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
        Returns:
            the output to the activation function
        '''

        return self.tf_activation(activations)


class L2Norm(Activation):
    '''a wrapper for an activation function that will add l2 normalisation'''

    def _apply_func(self, activations, is_training, reuse):
        '''
        apply own functionality
        Args:
            activations: the ioutputs to the wrapped activation function
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
        Returns:
            the output to the activation function
        '''

        with tf.variable_scope('l2_norm', reuse=reuse):
            #compute the mean squared value
            sig = tf.reduce_mean(tf.square(activations), 1, keep_dims=True)

            #divide the input by the mean squared value
            normalized = activations/sig

            #if the mean squared value is larger then one select the normalized
            #value otherwise select the unnormalised one
            return tf.select(tf.greater(tf.reshape(sig, [-1]), 1),
                             normalized, activations)

class Dropout(Activation):
    '''a wrapper for an activation function that will add dropout'''

    def __init__(self, activation, dropout):
        '''
        the Dropout_wrapper constructor
        Args:
            activation: the activation function being wrapped
            dropout: the dropout rate, has to be a value in (0:1]
        '''

        super(Dropout, self).__init__(activation)

        assert dropout > 0 and dropout <= 1
        self.dropout = dropout

    def _apply_func(self, activations, is_training, reuse):
        '''
        apply own functionality
        Args:
            activations: the ioutputs to the wrapped activation function
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
        Returns:
            the output to the activation function
        '''

        if is_training:
            return tf.nn.dropout(activations, self.dropout)
        else:
            return activations

class Batchnorm(Activation): # pylint: disable=too-few-public-methods
    '''A wrapper for an activation function that will add batch normalisation'''

    def _apply_func(self, activations, is_training, reuse):
        '''
        apply own functionality
        Args:
            activations: the ioutputs to the wrapped activation function
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
        Returns:
            the output to the activation function
        '''

        return tf.contrib.layers.batch_norm(activations,
                                            is_training=is_training,
                                            reuse=reuse, scope='batch_norm')
