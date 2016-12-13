'''@file layer.py
Neural network layers '''
from __future__ import absolute_import, division, print_function


import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

from IPython.core.debugger import Tracer; debug_here = Tracer()

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
            scope: the variable scope of the layer

        Returns:
            The output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):

                stddev = (self.weights_std if self.weights_std is not None
                          else 1/int(inputs.get_shape()[1])**0.5)

                weights = tf.get_variable(
                    'weights', [inputs.get_shape()[1], self.output_dim],
                    initializer=tf.random_normal_initializer(stddev=stddev))

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
    """This class allows enables blstm layer creation as well as computing
       their output. The output is found by linearly combining the forward
       and backward pass as described in:
       Graves et al., Speech recognition with deep recurrent neural networks,
       page 6646.
    """
    def __init__(self, num_units):
        """
        BlstmLayer constructor

        Args:
            num_units: The number of units in the LSTM
        """

        self.num_units = int(num_units)

    def __call__(self, inputs, sequence_length, is_training=False,
                 reuse=False, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                      for variables with the same name in the graph and reuse
                      these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell = rnn_cell.LSTMCell(self.num_units,
                                          state_is_tuple=True,
                                          use_peepholes=True)

            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell, lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(2, outputs_tupple)

            return outputs

class PLSTMLayer(object):
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

        self.num_units = int(num_units)
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


        with tf.variable_scope(scope or type(self).__name__, reuse=reuse) as scope:

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
    print('concat: initial shape: ', input_shape)
    concat_inputs = []
    for time_i in range(1, int(input_shape[1]), 2):
        concat_input = tf.concat(1, [inputs[:, time_i-1, :],
                                     inputs[:, time_i, :]],
                                 name='plstm_concat')
        concat_inputs.append(concat_input)

    inputs = tf.pack(concat_inputs, axis=1, name='plstm_pack')

    concat_shape = tf.Tensor.get_shape(inputs)
    print('concat: concat shape: ', concat_shape)


    sequence_lengths = tf.cast(tf.floor(tf.cast(sequence_lengths,
                                        tf.float32)/2),
                               tf.int32)
    return inputs, sequence_lengths



class Conv1dlayer(object):
    '''a 1-D convolutional layer'''

    def __init__(self, num_units, kernel_size, stride):
        '''constructor

        Args:
            num_units: the number of filters
            kernel_size: the size of the filters
            stride: the stride of the convolution
        '''

        self.num_units = num_units
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                      for variables with the same name in the graph and reuse
                      these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the outputs which is a [batch_size, max_length/stride, num_units]
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            stddev = 1/self.kernel_size**0.5
            input_dim = int(inputs.get_shape()[2])

            #the filte parameters
            w = tf.get_variable(
                'filter',
                [self.kernel_size, input_dim, self.num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #do the convolution
            out = tf.nn.conv1d(inputs, w, self.stride, padding = 'SAME')

        return out


class GatedDilatedConvolution(object):
    '''A gated dilated convolution layer'''
    def __init__(self, kernel_size):
        '''constructor

        Args:
            kernel_size: size of the filters
        '''

        self.kernel_size = kernel_size

    def __call__(self, inputs, dilation_rate, is_training=False, reuse=False,
                 scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            dilation_rate: the rate of dilation
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                      for variables with the same name in the graph and reuse
                      these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            a pair containing:
                - The residual output
                - the skip connections
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            num_units = int(inputs.get_shape()[2])
            stddev = 1/self.kernel_size**0.5

            #create the data filters
            wd = tf.get_variable(
                'data_filter', [1, self.kernel_size, num_units, num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #create the gate filters
            wg = tf.get_variable(
                'gate_filter', [1, self.kernel_size, num_units, num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #the weights of the linear layer
            wl = tf.get_variable(
                'linear', [1, num_units, num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #apply the dilated convolution
            out = tf.expand_dims(inputs, 1)
            out = tf.nn.tanh(tf.nn.atrous_conv2d(
                out, wd, dilation_rate, padding='SAME'))
            out = out[:,0,:,:]

            #compute the gate values
            gate = tf.expand_dims(inputs, 1)
            gate = tf.nn.sigmoid(tf.nn.atrous_conv2d(
                gate, wg, dilation_rate, padding='SAME'))
            gate = gate[:,0,:,:]

            #compute the gated output
            out = out*gate

            #compute the final output
            out = tf.nn.tanh(tf.nn.conv1d(out, wl, 1, padding='SAME'))

            #return the residual and the skip
            return inputs + out, out



