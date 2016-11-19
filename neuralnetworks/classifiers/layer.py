'''@file layer.py
Neural network layers '''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn import dynamic_rnn
from neuralnetworks.classifiers.activation import IdentityWrapper


class FFLayer(object):
    '''This class defines a fully connected feed forward layer'''

    def __init__(self, output_dim, activation, weights_std=None):
        '''
        FFLayer constructor, defines the variables
        Args:
            output_dim: output dimension of the layer
            activation: the activation function
            weights_std: the standart deviation of the weights by default the
                inverse square root of the input dimension is taken
        '''

        #save the parameters
        self.output_dim = output_dim
        self.activation = activation
        self.weights_std = weights_std

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        '''
        Create the variables and do the forward computation
        Args:
            inputs: the input to the layer
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the variable scope of the layer\
        Returns:
            The output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):

                if self.weights_std is not None:
                    stddev = self.weights_std
                else:
                    stddev = 1/int(inputs.get_shape()[1])**0.5
                weight_init = tf.random_normal_initializer(stddev=stddev)
                #weight_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

                weights = tf.get_variable(
                    'weights', [inputs.get_shape()[1], self.output_dim],
                    initializer=weight_init)

                biases = tf.get_variable(
                    'biases', [self.output_dim],
                    initializer=tf.constant_initializer(0))

            #apply weights and biases
            with tf.variable_scope('linear', reuse=reuse):
                linear = tf.matmul(inputs, weights) + biases

            #apply activation function
            with tf.variable_scope('activation', reuse=reuse):
                outputs = self.activation(linear, is_training, reuse)

        return outputs

class BLSTMLayer(object):
    """This class allows blstm layer creation as well as computing
       their output. The output is found by linearly combining the forward
       and backward pass as described in:
       Graves et al., Speech recognition with deep recurrent neural networks,
       page 6646.
    """

    def __init__(self, num_units, pyramidal=False):
        """
        BlstmLayer constructor
        Args:
            num_units: The number of units in the LSTM
            pyramidal: indicates if a pyramidal BLSTM is desired.
        """

        self.num_units = num_units
        self.pyramidal = pyramidal

    def __call__(self, inputs, sequence_lengths, is_training=False,
                 reuse=False, scope=None):
        """
        Create the variables and do the forward computation
        Args:
            inputs: A time minor tensor of shape [batch_size, time,
                    input_size],
            sequence_lengths: the length of the input sequences
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                      for variables with the same name in the graph and reuse
                      these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.
        Returns:
            the output of the layer, the concatenated outputs of the
            forward and backward pass shape [batch_size, time, input_size*2]
            or [batch_size, time/2, input_size*2] if self.plstm is set to
            true.
        """

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #create the lstm cell that will be used for the forward and backward
            lstm_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            lstm_cell = rnn_cell.LSTMCell(self.num_units,
                                          state_is_tuple=True,
                                          use_peepholes=True,
                                          initializer=lstm_init)

            if self.pyramidal is True:
                inputs, sequence_lengths = concatenate(inputs,
                                                       sequence_lengths,
                                                       scope)

            #outputs, output_states
            outputs, _ = bidirectional_dynamic_rnn(
                lstm_cell, lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_lengths)
            outputs = tf.concat(2, outputs)
        return outputs, sequence_lengths

def concatenate(inputs, sequence_lengths, scope):
    """
    Turn the blstm into a plstm by input concatinations.
    Args:
        inputs: A time minor tensor [batch_size, time, input_size]
        sequence_lengths: the length of the input sequences
        scope: the current scope
    Returns:
        inputs: Concatenated inputs [batch_size, time/2, input_size*2]
        sequence_lengths: the lengths of the inputs sequences [batch_size]
    """
    input_shape = tf.Tensor.get_shape(inputs)
    print(scope + ' initial shape: ', input_shape)
    concat_inputs = []
    for time_i in range(1, int(input_shape[1]), 2):
        concat_input = tf.concat(1, [inputs[:, time_i-1, :],
                                     inputs[:, time_i, :]],
                                 name='plstm_concat')
        concat_inputs.append(concat_input)

    inputs = tf.pack(concat_inputs, axis=1, name='plstm_pack')

    concat_shape = tf.Tensor.get_shape(inputs)
    print(scope + '  concat shape: ', concat_shape)

    sequence_lengths = tf.cast(tf.floor(sequence_lengths/2),
                               tf.int32)
    return inputs, sequence_lengths


class LinearCell(RNNCell):
    """
    A linear cell for the listener output layer
    """
    def __init__(self, output_dim, out_weights_std):
        self._output_dim = output_dim
        self._out_weights_std = out_weights_std
        identity_activation = IdentityWrapper()
        self.ff_layer = FFLayer(output_dim, identity_activation,
                                out_weights_std)

    @property
    def state_size(self):
        return 0

    @property
    def output_size(self):
        return self._output_dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            output = self.ff_layer(inputs)
        return output, state

class LinearLayer(object):
    """ This layer implements a sequential linear layer.
        The idea is to use a stateless linear Cell, in order
        to be able to dynamically unroll the linear layer computation,
        using the tensorflow default functions. """

    def __init__(self, output_dim, out_weights_std):
        self._output_dim = output_dim
        self._out_weights_std = out_weights_std

    def __call__(self, inputs, sequence_lengths, is_training=False,
                 reuse=False, scope=None):

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            linear_cell = LinearCell(self._output_dim, self._out_weights_std)
            outputs, _ = dynamic_rnn(linear_cell, inputs, dtype=tf.float32,
                                     sequence_length=sequence_lengths)

        return outputs

        