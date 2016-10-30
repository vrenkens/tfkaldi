'''@file layer.py
Neural network layers '''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn

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
            pyramidal: indicates if a pyramidal or p
        """

        self.num_units = num_units
        self.pyramidal = pyramidal

    def __call__(self, inputs, sequence_length, is_training=False,
                 reuse=False, scope=None):
        """
        Create the variables and do the forward computation
        Args:
            inputs: the input to the layer
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
            lstm_cell = rnn_cell.LSTMCell(self.num_units,
                                          state_is_tuple=True,
                                          use_peepholes=True)

            if self.pyramidal is True:
                print(scope + ' initial length ' + str(len(inputs)))
                print(scope + ' initial shape: ',
                      tf.Tensor.get_shape(inputs[0]))
                concat_inputs = []
                for time_i in range(1, len(inputs), 2):
                    concat_input = tf.concat(1, [inputs[time_i-1],
                                                 inputs[time_i]])
                    concat_inputs.append(concat_input)
                print(scope + ' concat length ' + str(len(concat_inputs)))
                print(scope + ' concat shape: ',
                      tf.Tensor.get_shape(concat_inputs[0]))
                inputs = concat_inputs

            #outputs, output_state_fw, output_state_bw
            outputs, _, _ = bidirectional_rnn(lstm_cell,
                                              lstm_cell,
                                              inputs, dtype=tf.float32)
                                              #sequence_length=sequence_length)
            #using the sequence_length argument causes memory
            #trouble sometimes.
        return outputs
