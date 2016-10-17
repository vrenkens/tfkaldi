import numpy as np
import tensorflow as tf
from copy import copy
from tensorflow.python.ops import rnn_cell
from neuralnetworks.nnet_layer import BlstmSettings
from neuralnetworks.nnet_layer import BlstmLayer
from neuralnetworks.nnet_layer import PyramidalBlstmLayer
from neuralnetworks.nnet_layer import FFLayer
from neuralnetworks.nnet_activations import TfWrapper
from custompython.lazy_decorator import lazy_property
from IPython.core.debugger import Tracer; debug_here = Tracer()

#disable the too few public methods complaint
# pylint: disable=R0903
#disable the print parenthesis warinig coming from python 2 pylint.
# pylint: disable=C0325

class Listener(object):
    """
    A set of pyramidal blstms, which compute high level audio features.
    """
    def __init__(self, blstm_settings, plstm_settings, plstm_layer_no,
                 output_dim):
        """ initialize the listener.
        """
        self.output_dim = output_dim
        #the Listerner foundation is a classical bidirectional Long Short
        #term mermory layer.
        blstm_settings.name = 'blstm_layer0'
        self.blstm_layer = BlstmLayer(blstm_settings)
        #on top of are three pyramidal BLSTM layers.
        self.plstms = []
        for layer_count in range(plstm_layer_no):
            plstm_settings.name = 'plstm_layer_' + str(layer_count)
            if (layer_count+1) == len(self.plstms):
                plstm_settings.output_dim = output_dim
            self.plstms.append(PyramidalBlstmLayer(plstm_settings))

    def __call__(self, input_features, sequence_lengths):
        """ Compute the output of the listener function. """
        #compute the base layer blstm output.
        hidden_values = self.blstm_layer(input_features, sequence_lengths)
        #move on to the plstm ouputs.
        for plstm_layer in self.plstms:
            hidden_values = plstm_layer(hidden_values)
        return hidden_values


class AttendAndSpell(object):
    """ Class implementing alignment establishment and transcription
        or the attend and spell part of the LAS-model.

    Internal Variables:
              features: (H)  the high level features the Listener computed.
         decoder_state: (s_i)
        context_vectors: (c_i) in the paper, found using the
                        attention_context function.
       character_probs: (y) output probability distribution.
    """
    def __init__(self, las_model):
        self.las_model = las_model
        with tf.variable_scope("attention"):

            self.feedforward_hidden_units = 42
            self.feedforward_hidden_layers = 4
            #the decoder state size must be equal to the RNN size.
            self.dec_state_size = 42

            #--------------------Create Variables-----------------------------#
            # setting up the decider_state, character distribution
            # and context vector variables.
            zero_initializer = tf.constant_initializer(value=0)
            self.decoder_state = tf.get_variable(
                                    name='current_decoder_state',
                                    shape=[self.dec_state_size, 1],
                                    initializer=zero_initializer,
                                    trainable=False)
            # the charater distirbution must initially be the sos token.
            # assuming encoding done as specified in the batch dispenser.
            # 0: ' ', 1: '<', 2:'>', ...
            # initialize to start of sentence token '<' or as one hot encoding:
            # 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            sos = np.zeros(self.las_model.target_label_no)
            sos[1] = 1
            sos_initializer = tf.constant_initializer(sos)
            self.char_dist_vec = tf.get_variable(
                                    name='char_dist_vec',
                                    shape=[self.las_model.target_label_no, 1],
                                    initializer=sos_initializer,
                                    trainable=False)
            # the dimension of the context vector is determined by the listener
            # output dimension.
            self.context_vector = tf.get_variable(
                                    name='context_vector',
                                    shape=[self.las_model.listen_output_dim,
                                           1],
                                        initializer=zero_initializer,
                                        trainable=False)

            self.scalar_energies = tf.get_variable(name='scalar_energies',
                        shape=[self.las_model.listen_output_dim,
                               1],
                        initializer=zero_initializer,
                        trainable=False)

            #--------------------Create network functions---------------------#
            # Feedforward layer custom parameters.
            activation = None
            activation = TfWrapper(activation, tf.nn.relu)


            state_net_dimension = FFNetDimension(self.dec_state_size,
                                                self.feedforward_hidden_units,
                                                self.feedforward_hidden_units,
                                                self.feedforward_hidden_layers)

            self.state_net = FeedForwardNetwork(state_net_dimension,
                                                activation)

            # copy the state net any layer settings
            # => all properties, which are not explicitly changed
            # stay the same.
            featr_net_dimension = copy(state_net_dimension)
            featr_net_dimension.input_dim = self.las_model.listen_output_dim
            self.featr_net = FeedForwardNetwork(featr_net_dimension,
                                                activation)

            self.decoder_rnn = RNN(self.dec_state_size, name='decoder_rnn')

            char_net_dimension = FFNetDimension(
                            input_dim=self.dec_state_size
                                      +self.las_model.listen_output_dim,
                            output_dim=self.las_model.target_label_no,
                            num_hidden_units=self.feedforward_hidden_units,
                            num_hidden_layers=self.feedforward_hidden_layers)

            self.char_net = FeedForwardNetwork(char_net_dimension,
                                               activation)

    def __call__(self, high_lvl_features):
        """
        Evaluate the attend and spell function in order to compute the
        desired character distribution.
        """

        with tf.variable_scope("attention_computation"):
            scalar_energy_lst = []
            state_list = self.decoder_rnn.get_zero_state_lst(
                                                    self.las_model.batch_size,
                                                    self.las_model.dtype)

            decoder_state = self.decoder_state
            char_dist_vec = self.char_dist_vec
            context_vector = self.context_vector

            for time, feat_vec in enumerate(high_lvl_features):
                debug_here()
                #TODO: fix input vector batch size collision problem
                # 137, 793...
                #TODO: Remove.

                #s_i = RNN(s_(i-1), y_(i-1), c_(i-1))
                rnn_input = tf.concat(0, [decoder_state,
                                       char_dist_vec,
                                       context_vector])
                decoder_state, state_list = self.decoder_rnn(rnn_input,
                                                             state_list)

                #compute the attention context.
                # e_(i,u) = psi(s_i)^T * phi(h_u)
                scalar_energy = tf.nnet.reduce_sum(
                                            self.featr_net(feat_vec)
                                           *self.state_net(decoder_state))
                scalar_energy_lst.append(scalar_energy)
                # alpha = softmax(e_(i,u))
                scalar_energy_tensor = tf.convert_to_tensor(scalar_energy_lst)
                alpha = tf.nn.softmax(scalar_energy_tensor)

                # find the context vector
                # c_i = sum(alpha*h_i)
                #compute the current context_vector assuming that vectors
                #ahead of the current time step do not matter.
                context_vector = 0*context_vector #set context_vector to zero.
                for t in range(0, time):
                    context_vector = (context_vector
                                    + alpha[t]*high_lvl_features[t])

                #construct the char_net input
                char_net_input = tf.concat(0, [decoder_state, context_vector])
                char_dist_vec = self.char_net(char_net_input)
                char_dist_vec = tf.nn.softmax(char_dist_vec)
        return char_dist_vec


class RNN(object):
    """
    Set up the RNN network which computes the decoder state.
    This function takes
    """
    def __init__(self, lstm_dim, name):
        self.layer_number = 2
        #create the two required LSTM blocks.
        self.blocks = []
        for i in range(0, self.layer_number):
            with tf.variable_scope(name + 'block' + str(i)):
                self.blocks.append(rnn_cell.LSTMCell(lstm_dim,
                                                   use_peepholes=True,
                                                   state_is_tuple=True))

    def get_zero_state_lst(self, batch_size, dtype):
        """ Get a list filled with zero states which can be used
            to start up the unrolled LSTM computations."""
        zero_state_list = []
        for block in self.blocks:
            zero_state_list.append(block.zero_state(batch_size, dtype))
        return zero_state_list

    def __call__(self, single_input, state_list):
        """
        Computes the RNN outputs for a single input. This CALL MUST BE
        UNROLLED MANUALLY.
        @param single_input: a single input vector containing
                             the output distribution and context
                             from the previous time step.
        @param state_list: a list containing the cell state
                           for each lstm in the block list.
        """

        assert(len(state_list) == len(self.blocks))
        inoutput = single_input
        new_states_list = []
        for idx, block in enumerate(self.blocks):
            inoutput, state = block(inoutput, state_list[idx])
            new_states_list.append(state)
        return inoutput, state_list


class FFNetDimension(object):
    """ Class containing the information to create Feedforward nets. """
    def __init__(self, input_dim, output_dim, num_hidden_units,
                 num_hidden_layers):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers

class FeedForwardNetwork(object):
    """ A class defining the feedforward MLP networks used to compute the
        scalar energy values required for the attention mechanism.
    """
    def __init__(self, dimension, activation):
        #store the settings
        self.dimension = dimension
        self.activation = activation

        #create the layers
        self.layers = [None]*(dimension.num_hidden_layers+1)
        #input layer
        self.layers[0] = FFLayer(dimension.num_hidden_units, activation)
        #hidden layers
        for k in range(1, len(self.layers)-1):
            self.layers[k] = FFLayer(dimension.num_hidden_units, activation)
        #output layer
        self.layers[-1] = FFLayer(dimension.output_dim, activation)

    def __call__(self, states_or_features):
        hidden = states_or_features
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden







