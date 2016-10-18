##@package nnetactivations
# contains activation functions

#pylint: disable=R0903
#abstract properties don't have to be overwritten they only have to be set
#pylint does not get that.
#pylint: disable=W0223
# disable python 2 bracket warning.
#pylint: disable=C0325

import tensorflow as tf
from abc import ABCMeta, abstractmethod

class Activation(object):
    """an activation function interface class"""
    __metaclass__ = ABCMeta

    def __init__(self, activation):
        '''make sure everyone has their activations set.
           which is guarenteed if they use the meta-constructor.'''
        self.activation = activation

    def __call__(self, inputs, is_training=False, reuse=False):
        """apply the activation function
        @param inputs the inputs to the activation function
        @param is_training is_training whether or not the network is
               in training mode
        @param reuse wheter or not the variables in the network
                should be reused

        @return the output to the activation function
        """

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
        """
        Apply a custom function.
        @param activations the outputs to the wrapped activation function
        @param is_training is_training whether or not the network
               is in training mode
        @param reuse wheter or not the variables in the network
               should be reused

        @return the output to the activation function
        """
        raise NotImplementedError("Abstract method")

class TfWrapper(Activation):
    """ a wrapper for an activation function that will add
        a tf activation function
    """

    def __init__(self, activation, tf_activation):
        """
        the TfWrapper constructor

        @param activation the activation function being wrapped
        @param the tensorflow activation function that is wrapping
        """
        super(TfWrapper, self).__init__(activation)
        self.tf_activation = tf_activation

    def _apply_func(self, activations, is_training, reuse):
        """
        apply own functionality
        #
        #@param activations the ioutputs to the wrapped activation function
        #@param is_training is_training whether or not the network is in
                training mode
        #@param reuse wheter or not the variables in the network should be
                reused
        #
        #@return the output to the activation function
        """
        return self.tf_activation(activations)



class L2Wrapper(Activation):
    """ a wrapper for an activation function that will add l2 normalisation """
    def __init__(self, activation):
        """ the L2_wrapper constructor
        @param activation the activation function being wrapped
        """
        super(L2Wrapper, self).__init__(activation)

    def _apply_func(self, activations, is_training, reuse):
        """
        apply own functionality
        @param activations the ioutputs to the wrapped activation function
        @param is_training is_training whether or not the network is in training
         mode
        @param reuse wheter or not the variables in the network should be reused
        @return the output to the activation function
        """
        with tf.variable_scope('l2_norm', reuse=reuse):
            #compute the mean squared value
            sig = tf.reduce_mean(tf.square(activations), 1, keep_dims=True)

            #divide the input by the mean squared value
            normalized = activations/sig

            #if the mean squared value is larger then one select the
            # normalized value otherwise select the unnormalised one
            return tf.select(tf.greater(tf.reshape(sig, [-1]), 1),
            							normalized, activations)

class DropoutWrapper(Activation):
    """  a wrapper for an activation function that will add dropout """
    def __init__(self, activation, dropout):
        """ the Dropout_wrapper constructor
        @param activation the activation function being wrapped
        @param dopout the dropout rate, has to be a value in (0:1]
        """
        super(DropoutWrapper, self).__init__(activation)
        assert(dropout > 0 and dropout <= 1)
        self.dropout = dropout

    def _apply_func(self, activations, is_training, reuse):
        """apply own functionality
        @param activations the ioutputs to the wrapped activation function
        @param is_training is_training whether or not the network is in
                            training mode
        @param reuse wheter or not the variables in the network should be
               reused
        @return the output to the activation function
        """

        if is_training:
            return tf.nn.dropout(activations, self.dropout)
        else:
            return activations

class BatchnormWrapper(Activation):
    """a wrapper for an activation function
       that will add batch normalisation.
    """

    def __init__(self, activation):
        """ the Batchnorm_wrapper constructor
        @param activation the activation function being wrapped
        """
        super(BatchnormWrapper, self).__init__(activation)
        self.activation = activation

    def _apply_func(self, activations, is_training, reuse):
        """apply own functionality
        @param activations the ioutputs to the wrapped activation function
        @param is_training is_training whether or not the network
               is in training mode
        @param reuse wheter or not the variables in the network
               should be reused
        @return the output to the activation function
        """
        return tf.contrib.layers.batch_norm(activations,
                                            is_training=is_training,
                                            reuse=reuse,
                                            scope='batch_norm')
