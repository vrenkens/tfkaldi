'''@package nnetlayer
# contains neural network layers
'''

import tensorflow as tf
import numpy as np
from copy import copy
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
from IPython.core.debugger import Tracer; debug_here = Tracer()


# fix the pylint state is tuple error
# pylint: disable=E1123
# disable the too few public methods complaint
# pylint: disable=R0903
# disable the print parenthesis warinig coming from python 2 pylint.
# pylint: disable=C0325

class BlstmSettings(object):
    ''' An parameter class grouping parameters required to create
        lstm layers.'''
    def __init__(self, output_dim, lstm_dim, weights_std, name):
        self.output_dim = output_dim
        self.lstm_dim = lstm_dim
        self.weights_std = weights_std
        self.name = name

##This class defines a bidirectional LSTM layer.
class BlstmLayer(object):
    '''This class allows enables blstm layer creation as well as computing
       their output. The output is found by linearly combining the forward
       and backward pass as described in:
       Graves et al., Speech recognition with deep recurrent neural networks,
       page 6646.
    '''

    def __init__(self, settings):
        '''
        Bidirectional LSTM-Layer constructor, defines the variables
        the parameters for the BLSTM must be passed with a BlstmSettings
        object.
        @param settings a BlstmSettings object containing
               all information required to create BLSTM layers.
        '''
        self.name = settings.name
        with tf.variable_scope(settings.name + '_forward'):
            self.forward_lstm_block = rnn_cell.LSTMCell(settings.lstm_dim,
                                                        use_peepholes=True,
                                                        state_is_tuple=True)
        with tf.variable_scope(settings.name + '_backward'):
            self.backward_lstm_block = rnn_cell.LSTMCell(settings.lstm_dim,
                                                         use_peepholes=True,
                                                         state_is_tuple=True)

        #create the model parameters in this layer
        with tf.variable_scope(settings.name + '_parameters'):
            weight_init = tf.random_normal_initializer(stddev=
                                                    settings.weights_std)
            self.weights = tf.get_variable('weights', [2*settings.lstm_dim,
                                                        settings.output_dim],
                                                       initializer=weight_init)

            self.biases = tf.get_variable('biases', [settings.output_dim],
                                       initializer=tf.constant_initializer(0))

    def __call__(self, inputs, sequence_length):
        with tf.name_scope(self.name + '_call') as scope:
            #outputs, output_state_fw, output_state_bw
            outputs, _, _ = bidirectional_rnn(self.forward_lstm_block,
                                              self.backward_lstm_block,
                                              inputs, dtype=tf.float32,
                                              scope=scope)
                #sequence_length=sequence_length)
                #using the sequence_length argument causes memory
                #trouble sometimes.
            #output size: [time][batch][cell_fw.output_size
            #                           +cell_bw.output_size]
            #linear neuron computes the output for loop loops trought time.
            blstm_logits = [tf.matmul(T, self.weights) + self.biases
                            for T in outputs]
            #lotis shape [max_time_steps][batch_size, output_dim]

            return blstm_logits


class PyramidalBlstmLayer(BlstmLayer):
    '''
    Bidirectional LSTM-Layer constructor, defines the variables
    output_dim = number of classes.
    See Listen, attend and spell Chan et al:
    typical   BLSTM: h[i,j] = BLSTM(h[i-1,j], h[i,j-1])
    pyramidal BLSTM: h[i,j] = BLSTM(h[i-1,j], [h[2i, j-1] h[2i+1, j-1]]);
    in other words the pyramidal BLSTM cocatenates the two vectors
    from the previous layer in time.
    #TODO: Merge with with BlstmLayer.
    '''
    def __call__(self, inputs):
        with tf.name_scope(self.name + '_call') as scope:
            #concatenate in time within the input layer
            #this input comes from the previous layer
            #so we have j-1=const for the input list.
            print(self.name + ' initial length ' + str(len(inputs)))
            print(self.name + ' initial shape: ',
                    tf.Tensor.get_shape(inputs[0]))
            concat_inputs = []
            for time_i in range(1, len(inputs), 2):
                concat_input = tf.concat(1, [inputs[time_i-1], inputs[time_i]])
                concat_inputs.append(concat_input)
            print(self.name + ' concat length ' + str(len(concat_inputs)))
            print(self.name + ' concat shape: ',
                    tf.Tensor.get_shape(concat_inputs[0]))

            outputs, _, _ = bidirectional_rnn(self.forward_lstm_block,
                                              self.backward_lstm_block,
                                              concat_inputs, dtype=tf.float32,
                                              scope=scope)
            #output size: [time][batch][cell_fw.output_size
            #                           + cell_bw.output_size]
            #linear neuron computes the output for loop loops trought time.
            blstm_logits = [tf.matmul(T, self.weights) + self.biases
                            for T in outputs]
            #lotis shape [max_time_steps][batch_size, output_dim]
            return blstm_logits


class FFLayer(object):
    '''
    This class defines a fully connected feed forward layer
    '''

    def __init__(self, output_dim, activation, weights_std=None):
        '''
        FFLayer constructor, initializes the variables

        @param output_dim output dimension of the layer
        @param activation the activation function
        @param weights_std the standart deviation of the weights by default the
         inverse square root of the input dimension is taken
        '''
        self.output_dim = output_dim
        self.activation = activation
        self.weights_std = weights_std
        if self.weights_std is not None:
            self.initializer = tf.random_normal_initializer(
                                        stddev=self.weights_std)
        else:
            self.initializer = None

    def __call__(self, inputs, is_training=False,
                 reuse=False, scope=None):
        '''
        Do the forward computation

        @param inputs the input to the layer
        @param is_training is_training whether or not the network
               is in training mode
        @param reuse wheter or not the variables in the network should be reused
        @param scope the variable scope of the layer

        @return the output of the layer and the training output of the layer
        '''
        if self.initializer == None:
            self.initializer = 1/int(inputs.get_shape()[1])**0.5

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):
                weights = tf.get_variable('weights',
                                         [inputs.get_shape()[1],
                                          self.output_dim],
                                          initializer=self.initializer)
                biases = tf.get_variable('biases', [self.output_dim],
                                     initializer=tf.constant_initializer(0))

            #apply weights and biases
            with tf.variable_scope('linear', reuse=reuse):
                linear = tf.matmul(inputs, weights) + biases

            #apply activation function
            with tf.variable_scope('activation', reuse=reuse):
                outputs = self.activation(linear, is_training, reuse)
        return outputs
