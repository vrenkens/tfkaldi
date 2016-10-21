'''This module contains the elements needed to set up the listen attend and
spell network.'''

from copy import copy
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

# we are currenly in neuralnetworks, add it to the path.
sys.path.append("neuralnetworks")
from nnet_layer import BlstmLayer
from nnet_layer import PyramidalBlstmLayer
from nnet_layer import FFLayer
from nnet_activations import TfWrapper

from IPython.core.debugger import Tracer
debug_here = Tracer()

#disable the too few public methods complaint
# pylint: disable=R0903

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
        self.high_lvl_features = None
        self.las_model = las_model
        with tf.variable_scope("attention"):
            self.feedforward_hidden_units = 42
            self.feedforward_hidden_layers = 4
            #the decoder state size must be equal to the RNN size.
            self.dec_state_size = 42

            #TODO: Do this better.
            self.eos_treshold = tf.constant(0.8)
            self.max_target_length = tf.constant(171)

            #--------------------Create Variables-----------------------------#
            # setting up the decider_state, character distribution
            # and context vector variables.
            zero_initializer = tf.constant_initializer(value=0)
            self.decoder_state = tf.get_variable(
                name='current_decoder_state',
                shape=[self.las_model.batch_size, self.dec_state_size],
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
                shape=[self.las_model.batch_size,
                       self.las_model.target_label_no],
                initializer=sos_initializer,
                trainable=False)
            # the dimension of the context vector is determined by the listener
            # output dimension.
            self.context_vector = tf.get_variable(
                name='context_vector',
                shape=[self.las_model.batch_size,
                       self.las_model.listen_output_dim],
                initializer=zero_initializer,
                trainable=False)

            self.scalar_energies = tf.get_variable(
                name='scalar_energies',
                shape=[self.las_model.listen_output_dim, 1],
                initializer=zero_initializer,
                trainable=False)

            self.char_dist_tensor = tf.get_variable(
                name='char_dist_tensor',
                shape=[1, self.las_model.batch_size,
                       self.las_model.target_label_no],
                initializer=sos_initializer,
                trainable=False)

            #--------------------Create network functions---------------------#
            # Feedforward layer custom parameters.
            activation = None
            activation = TfWrapper(activation, tf.nn.relu)


            state_net_dimension = FFNetDimension(self.dec_state_size,
                                                 self.feedforward_hidden_units,
                                                 self.feedforward_hidden_units,
                                                 self.feedforward_hidden_layers
                                                )

            self.state_net = FeedForwardNetwork(state_net_dimension,
                                                activation, name='state_net')

            # copy the state net any layer settings
            # => all properties, which are not explicitly changed
            # stay the same.
            featr_net_dimension = copy(state_net_dimension)
            featr_net_dimension.input_dim = self.las_model.listen_output_dim
            self.featr_net = FeedForwardNetwork(featr_net_dimension,
                                                activation, name='featr_net')

            self.decoder_rnn = RNN(self.dec_state_size, name='decoder_rnn')

            char_net_dimension = FFNetDimension(
                input_dim=self.dec_state_size
                +         self.las_model.listen_output_dim,
                output_dim=self.las_model.target_label_no,
                num_hidden_units=self.feedforward_hidden_units,
                num_hidden_layers=self.feedforward_hidden_layers)

            self.char_net = FeedForwardNetwork(char_net_dimension,
                                               activation,
                                               name='char_net')


    def __call__(self, high_lvl_features):
        """
        Evaluate the attend and spell function in order to compute the
        desired character distribution.
        """
        self.high_lvl_features = high_lvl_features
        with tf.variable_scope("attention_computation"):
            rnn_states = self.decoder_rnn.get_zero_states(
                self.las_model.batch_size,
                self.las_model.dtype)
            decoder_state = self.decoder_state
            context_vector = self.context_vector
            char_dist_vec = self.char_dist_vec
            char_dist_tensor = self.char_dist_tensor

            i = tf.get_variable('loop_counter', [], dtype=tf.int32
                                , initializer=tf.constant_initializer(0))
            loop_vars = (i, decoder_state, context_vector, char_dist_vec,
                         char_dist_tensor, rnn_states)

            # set up the invariants. Always take the vecor shapes,
            # instead of the first example, where the first dimension is set
            # to to vary, because the while function body concatenates new
            # values into the first dimension.
            invariants = (i.get_shape(),
                          decoder_state.get_shape(),
                          context_vector.get_shape(),
                          char_dist_vec.get_shape(),
                          tf.TensorShape([None,
                                          self.las_model.batch_size,
                                          self.las_model.target_label_no]),
                          rnn_states.get_shape())

            loop_vars = tf.while_loop(self.cond, self.body,
                                      loop_vars=[loop_vars],
                                      shape_invariants=[invariants],
                                      parallel_iterations=1,
                                      name="attend_and_spell_loop")

            i, decoder_state, context_vector, char_dist_vec, \
                char_dist_tensor, rnn_states = loop_vars[0]

            print("char_dist_tensor shape",
                  tf.Tensor.get_shape(char_dist_tensor))
            return char_dist_tensor


    def cond(self, loop_vars):
        ''' Condition in charge of the attend and spell
            while loop. It checks if all the spellers in the current batch
            are confident of having found an eos token or if the maximum
            sequence length we expect to see in the data has been reached. '''
        #the encoding table has the eos token ">" placed at position 0.
        #i.e. " ", "<", ">", ...

        #pylint: disable=W0612
        #pylint does not know that the loop_vars are fixed and cannot
        #       all be used in this function.
        i, decoder_state, context_vector, char_dist_vec, \
            char_dist_tensor, rnn_states = loop_vars

        eos_prob = char_dist_vec[:, 0]
        loop_continue_conditions = tf.logical_and(
            tf.less(eos_prob, self.eos_treshold),
            tf.less(i, self.max_target_length))

        loop_continue_counter = tf.reduce_sum(tf.to_int32(
            loop_continue_conditions))
        keep_working = tf.not_equal(loop_continue_counter, 0)
        return keep_working

    def body(self, loop_vars):
        '''
        The whole loop body performing the attend and spell computations.
        '''
        i, decoder_state, context_vector, char_dist_vec, \
            char_dist_tensor, rnn_states = loop_vars
        high_lvl_features = self.high_lvl_features

        #s_i = RNN(s_(i-1), y_(i-1), c_(i-1))
        #Dimensions:   decoder_state_size , 42
        #            + alphabet_size,       31
        #            + listener_output_dim, 64
        #                                   137
        rnn_input = tf.concat(1, [decoder_state,
                                  char_dist_vec,
                                  context_vector])
        #TODO: Check this. The papers speak of the state not of the
        #      output. Is there a clean way to use the state
        #      of multilayered RNN?
        decoder_state, rnn_states = self.decoder_rnn(rnn_input,
                                                     rnn_states)

        #for loop runing trough the high level features.
        #assert len(high level features) == U.
        scalar_energy_lst = []
        for feat_vec in high_lvl_features:
            ### compute the attention context. ###
            # e_(i,u) = psi(s_i)^T * phi(h_u)
            phi = self.state_net(decoder_state)
            psi = self.featr_net(feat_vec)
            scalar_energy = tf.reduce_sum(psi*phi, reduction_indices=1,
                                          name='dot_sum')

            scalar_energy_lst.append(scalar_energy)
        # alpha = softmax(e_(i,u))
        scalar_energy_tensor = tf.convert_to_tensor(
            scalar_energy_lst)
        #Alpha has the same shape as the scalar_energy_tensor
        alpha = tf.nn.softmax(scalar_energy_tensor)

        ### find the context vector. ###
        # c_i = sum(alpha*h_i)
        context_vector = 0*context_vector
        for t in range(0, len(high_lvl_features)):
            #reshaping from (batch_size,) to (batch_size,1) is
            #needed for broadcasting.
            current_alpha = tf.reshape(alpha[t, :],
                                       [self.las_model.batch_size,
                                        1])
            context_vector = (context_vector
                              + current_alpha*high_lvl_features[t])

        #construct the char_net input
        char_net_input = tf.concat(1, [decoder_state, context_vector])
        logits = self.char_net(char_net_input)

        # TODO: tf.exp(logits[:,0]) / tf.reduce_sum(tf.exp(logits))
        # question: can the RNN input take the logits instead of
        #           the normalized char_dist_vec?
        char_dist_vec = tf.nn.softmax(logits)

        debug_here()

        char_dist_vec_shape = tf.Tensor.get_shape(char_dist_vec).as_list()
        #add a new dimension for concatenenation.
        char_dist_vec_conc = tf.reshape(char_dist_vec, [1,
                                                        char_dist_vec_shape[0],
                                                        char_dist_vec_shape[1]]
                                       )
        #concatenate the new result in the first (time) dimension.
        char_dist_tensor = tf.concat(0, [char_dist_tensor, char_dist_vec_conc])

        loop_vars = (i, decoder_state, context_vector, char_dist_vec,
                     char_dist_tensor, rnn_states)
        return [loop_vars]


class RNN(object):
    """
    Set up the RNN network which computes the decoder state.
    """
    def __init__(self, lstm_dim, name):
        self.name = name
        self.layer_number = 2
        #create the two required LSTM blocks.
        self.blocks = []
        for _ in range(0, self.layer_number):
            self.blocks.append(rnn_cell.LSTMCell(lstm_dim,
                                                 use_peepholes=True,
                                                 state_is_tuple=False))
        self.wrapped_cells = rnn_cell.MultiRNNCell(self.blocks,
                                                   state_is_tuple=False)
        self.reuse = None

    def get_zero_states(self, batch_size, dtype):
        """ Get a list filled with zero states which can be used
            to start up the unrolled LSTM computations."""
        return self.wrapped_cells.zero_state(batch_size, dtype)

    def __call__(self, single_input, state):
        """
        Computes the RNN outputs for a single input. This CALL MUST BE
        UNROLLED MANUALLY.
        """
        #assertion only works if state_is_touple is set to true.
        #assert len(state) == len(self.blocks)

        with tf.variable_scope(self.name + '_call', reuse=self.reuse):
            output = self.wrapped_cells(single_input, state)

        if self.reuse is None:
            self.reuse = True

        return output

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
    def __init__(self, dimension, activation, name):
        #store the settings
        self.dimension = dimension
        self.activation = activation
        self.name = name
        self.reuse = None

        #create the layers
        self.layers = [None]*(dimension.num_hidden_layers + 1)
        #input layer
        self.layers[0] = FFLayer(dimension.num_hidden_units, activation)
        #hidden layers
        for k in range(1, len(self.layers)-1):
            self.layers[k] = FFLayer(dimension.num_hidden_units, activation)
        #output layer
        self.layers[-1] = FFLayer(dimension.output_dim, activation)

    def __call__(self, states_or_features):
        hidden = states_or_features
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, scope=(self.name + '/' + str(i)),
                           reuse=(self.reuse))
        #set reuse to true after the variables have been created in the first
        #call.
        if self.reuse is None:
            self.reuse = True
        return hidden

def interval_print(interval, time, string, value):
    '''Print vector shapes only every interval time steps.'''
    if time%interval == 0:
        print(string, value)


