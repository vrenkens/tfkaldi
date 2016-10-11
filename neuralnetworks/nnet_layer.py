'''@package nnetlayer
# contains neural network layers
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
from IPython.core.debugger import Tracer; debug_here = Tracer()

# fix the pylint state is tuple error
# pylint: disable=E1123
#disable the too few public methods complaint
# pylint: disable=R0903
#disable the print parenthesis warinig coming from python 2 pylint.
# pylint: disable=C0325

class BlstmSettings(object):
    ''' An argument class grouping lstm related parameters '''
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



##This class defines a fully connected feed forward layer
class FFLayer(object):
    '''
    FFLayer constructor, defines the variables.

    @param input_dim input dimension of the layer.
    @param output_dim output dimension of the layer.
    @param weights_std standard deviation of the weights initializer.
    @param name name of the layer.
    @param transfername name of the transfer function that is used.
    @param l2_norm boolean that determines of l2_normalisation
           is used after every layer.
    @param dropout the chance that a hidden unit is propagated to the
           next layer.
    '''

    def __init__(self, input_dim, output_dim, weights_std, name,
         transfername='linear', l2_norm=False, dropout=1):

        #create the model parameters in this layer
        with tf.variable_scope(name + '_parameters'):
            self.weights = tf.get_variable('weights', [input_dim, output_dim],
                initializer=tf.random_normal_initializer(stddev=weights_std))
            self.biases = tf.get_variable('biases', [output_dim],
                initializer=tf.constant_initializer(0))

        #save the parameters
        self.transfername = transfername
        self.l2_norm = l2_norm
        self.dropout = dropout
        self.name = name

    ## Do the forward computation.
    #
    #@param inputs the input to the layer
    #@param apply_dropout bool to determine if dropout is aplied
    #
    #@return the output of the layer
    def __call__(self, inputs, apply_dropout=True):

        with tf.name_scope(self.name +'_call'):

            #apply weights and biases
            outputs = transfer_function(tf.matmul(inputs, self.weights) +
             self.biases, self.transfername)

            #apply l2 normalisation
            if self.l2_norm:
                outputs = transfer_function(outputs, 'l2_norm')

            #apply dropout
            if self.dropout < 1 and apply_dropout:
                outputs = tf.nn.dropout(outputs, self.dropout)

        return outputs



def transfer_function(inputs, name):
    '''
    ##Apply the transfer function
    #
    #@param inputs the inputs to the transfer function
    #@param name the name of the function, current options are: relu, sigmoid,
    #       tanh, linear or l2_norm
    #
    #@return the output to the transfer function
    '''
    if name == 'relu':
        return tf.nn.relu(inputs)
    elif name == 'sigmoid':
        return tf.nn.sigmoid(inputs)
    elif name == 'tanh':
        return tf.nn.tanh(inputs)
    elif name == 'linear':
        return inputs
    elif name == 'l2_norm':
        with tf.name_scope('l2_norm'):
            #compute the mean squared value
            sig = tf.reduce_mean(tf.square(inputs), 1, keep_dims=True)

            #divide the input by the mean squared value
            normalized = inputs/sig

            #if the mean squared value is larger then one select the normalized
            # value otherwise select the unnormalised one
            return tf.select(tf.greater(tf.reshape(sig, [-1]), 1), normalized,
             inputs)
    else:
        raise Exception('unknown transfer function: %s' % name)
