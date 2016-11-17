'''This module contains the elements needed to set up the listen attend and
spell network.'''

from copy import copy
import sys
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.util import nest

# we are currenly in neuralnetworks, add it to the path.
from neuralnetworks.classifiers.layer import FFLayer
from neuralnetworks.classifiers.layer import BLSTMLayer
from neuralnetworks.classifiers.activation import TfActivation
from neuralnetworks.classifiers.activation import IdentityWrapper

from IPython.core.debugger import Tracer; debug_here = Tracer();

#disable the too few public methods complaint
# pylint: disable=R0903


class Listener(object):
    """
    A set of pyramidal blstms, which compute high level audio features.
    """
    def __init__(self, lstm_dim, plstm_layer_no, output_dim, out_weights_std):
        """ Initialize the listener.
        """
        self.output_dim = output_dim
        self.lstm_dim = lstm_dim
        self.plstm_layer_no = plstm_layer_no
        self.output_dim = output_dim

        #the listener foundation is a classical bidirectional Long Short
        #term memory layer.
        self.blstm_layer = BLSTMLayer(lstm_dim, pyramidal=False)
        #on top of are three pyramidal BLSTM layers.
        self.plstm_layer = BLSTMLayer(lstm_dim, pyramidal=True)

        identity_activation = IdentityWrapper()
        self.output_layer = FFLayer(output_dim, identity_activation,
                                    out_weights_std)
        self.reuse = None

    def __call__(self, input_features, sequence_lengths):
        """ Compute the output of the listener function. """
        #compute the base layer blstm output.
        with tf.variable_scope(type(self).__name__, reuse=self.reuse) as scope:
            hidden_values, sequence_lengths = \
                self.blstm_layer(input_features,
                                 sequence_lengths,
                                 reuse=self.reuse,
                                 scope=("blstm_layer"))
            #move on to the plstm outputs.
            for counter in range(self.plstm_layer_no):
                hidden_values, sequence_lengths = \
                    self.plstm_layer(hidden_values,
                                     sequence_lengths,
                                     reuse=self.reuse,
                                     scope=("plstm_layer_" + str(counter)))
            output_values = self.linear_output_layer(hidden_values, scope)
        if self.reuse is None:
            self.reuse = True
        return output_values, sequence_lengths

    def linear_output_layer(self, hidden_values, scope):
        """ Run the concatenated forward and backward layer computations trough
            a simple linear output layer as described in:
            Speech recognition with deep recurrent neural networks,
            Graves, A, Mohamed, A.-R., Hinton, G
        """
        reuse = None or self.reuse
        scope = type(self).__name__ + "linear_layer"
        with tf.variable_scope(scope, reuse=reuse):
            hidden_time = tf.Tensor.get_shape(hidden_values)[1]
            output_values_lst = []
            #TODO: Test.
            for time in range(int(hidden_time)):
                output_value = hidden_values[:, time, :]
                output_values_lst.append(self.output_layer(
                    output_value, reuse=reuse, scope=scope))
                if reuse is None:
                    reuse = True
            output_values = tf.pack(output_values_lst, axis=1)
        return output_values

#create a tf style cell state tuple object to derive the actual tuple from.
_AttendAndSpellStateTouple = \
    collections.namedtuple(
        "AttendAndSpellStateTouple",
        "pre_context_states, post_context_states, one_hot_char, context_vector"
        )
class StateTouple(_AttendAndSpellStateTouple):
    """ Tuple used by Attend and spell cells for `state_size`,
     `zero_state`, and output state.
      Stores four elements:
      `(pre_context_states, post_context_states, one_hot_char,
            context_vector)`, in that order.
    """
    @property
    def dtype(self):
        """ Check if the all internal state variables have the same data-type
            if yes return that type. """
        for i in range(1, len(self)):
            if self[i-1].dtype != self[i].dtype:
                raise TypeError("Inconsistent internal state: %s vs %s" %
                                (str(self[i-1].dtype), str(self[i].dtype)))
        return self[0].dtype

    @property
    def _shape(self):
        """ Make shure tf.Tensor.get_shape(this) returns  the correct output.
        """
        return self.get_shape()

    def get_shape(self):
        """ Return the shapes of the elements contained in the state tuple. """
        flat_shapes = []
        flat_self = nest.flatten(self)
        for i in range(0, len(flat_self)):
            flat_shapes.append(tf.Tensor.get_shape(flat_self[i]))
        shapes = nest.pack_sequence_as(self, flat_shapes)
        return shapes


class AttendAndSpellCell(RNNCell):
    """
    Define an attend and Spell Cell. This cell takes the high level features
    as input. During training the groundtruth values are fed into the network
    as well.

    Internal Variables:
              features: (H) the high level features the Listener computed.
         decoder_state: (s_i) ambiguous in the las paper split in two here.
       context_vectors: (c_i) in the paper, found using the
                        attention_context function.
          one_hot_char: (y) one hot encoded input and output char.
    """
    def __init__(self, las_model, decoder_state_size=40,
                 feedforward_hidden_units=56, feedforward_hidden_layers=3):
        self.feedforward_hidden_units = feedforward_hidden_units
        self.feedforward_hidden_layers = feedforward_hidden_layers
        #the decoder state size must be equal to the RNN size.
        self.dec_state_size = decoder_state_size
        self.high_lvl_features = None
        self.las_model = las_model

        #Determines whether the post_context_rnn will be used.
        self.type_two = True

        #--------------------Create network functions-------------------------#
        # TODO: Move outside the cell
        # Feed-forward layer custom parameters. Vincent knows more about these.
        activation = None
        activation = TfActivation(activation, tf.nn.relu)

        state_net_dimension = FFNetDimension(self.dec_state_size,
                                             self.feedforward_hidden_units,
                                             self.feedforward_hidden_units,
                                             self.feedforward_hidden_layers)

        self.state_net = FeedForwardNetwork(state_net_dimension,
                                            activation, name='state_net')
        # copy the state net any layer settings
        # => all properties, which are not explicitly changed
        # stay the same.
        featr_net_dimension = copy(state_net_dimension)
        self.featr_net = FeedForwardNetwork(featr_net_dimension,
                                            activation, name='featr_net')

        self.pre_context_rnn = RNN(self.dec_state_size,
                                   name='pre_context_rnn')
        self.post_context_rnn = RNN(self.dec_state_size,
                                    name='post_context_rnn')

        char_net_dimension = FFNetDimension(
            input_dim=self.dec_state_size
            +         self.las_model.listener.output_dim,
            output_dim=self.las_model.target_label_no,
            num_hidden_units=self.feedforward_hidden_units,
            num_hidden_layers=self.feedforward_hidden_layers)

        self.char_net = FeedForwardNetwork(char_net_dimension,
                                           activation,
                                           name='char_net')

        #---------------------One hot char selection variables----------------#
        self.ones = None
        self.one_hot_char_shape = None


    def set_features(self, high_lvl_features):
        ''' Set the features when available, storing the features in the
            object makes the cell call simpler.'''
        print("     Feature dimension:",
              tf.Tensor.get_shape(high_lvl_features))
        self.high_lvl_features = tf.unpack(high_lvl_features, axis=1)


    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell.
        """
        return self.las_model.target_label_no

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer,
        a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return self.zero_state(self.las_model.batch_size, self.las_model.dtype).get_shape()

    def zero_state(self, batch_size, dtype):
        """Return an initial state for the Attend and state cell.
            @returns an StateTouple object filled with the state variables.
        """
        #the batch_size has to be fixed in order to be able to corretly
        #return the state_sizes, should self.state_size() be called before
        #the zero states are created.
        assert batch_size == self.las_model.batch_size
        assert dtype == self.las_model.dtype

        zero_state_scope = type(self).__name__+ "_zero_state"
        with tf.variable_scope(zero_state_scope):
            #----------------------Create Zero state tensors---------------------------#
            # setting up the decoder_RNN_states, character distribution
            # and context vector variables.
            pre_context_states = self.pre_context_rnn.get_zero_states(
                batch_size, dtype)
            post_context_states = self.post_context_rnn.get_zero_states(
                batch_size, dtype)

            # The character distribution must initially be the sos token.
            # assuming encoding done as specified in the batch dispenser.
            # 0: '>', 1: '<', 2:' ', ...
            # initialize to start of sentence token '<' as one hot encoding:
            # 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            sos_np = np.ones(batch_size, dtype=np.int32)
            sos = tf.constant(sos_np, tf.int32, shape=[batch_size])
            one_hot_char = tf.one_hot(sos, self.las_model.target_label_no,
                                      axis=1, dtype=dtype)
            print("    one_hot_char shape:", tf.Tensor.get_shape(one_hot_char))
            # The dimension of the context vector is determined by the listener
            # output dimension.
            zero_context_np = np.zeros([batch_size,
                                        self.las_model.listener.output_dim])
            context_vector = tf.constant(zero_context_np, dtype) 
            context_vector = tf.identity(context_vector)
            print("  context_vector shape:",
                  tf.Tensor.get_shape(context_vector))

        return StateTouple(pre_context_states, post_context_states,
                           one_hot_char, context_vector)


    def __call__(self, cell_input, state, scope=None, reuse=False):
        """
        Do the computations for a single unrolling of the attend and
        spell network.
        During training make sure training_char_input contains
        valid groundtrouth values.
        """
        as_cell_call_scope = scope or (type(self).__name__+ "_call")
        with tf.variable_scope(as_cell_call_scope, reuse=reuse):
            groundtruth_char = cell_input
            # pylint: disable = E0633
            # pylint does not know that StateTouple extends a collection
            # data type.
            pre_context_states, post_context_states, one_hot_char, \
                context_vector = state

            if self.high_lvl_features is None:
                raise AttributeError("Features must be set.")

            #Pick the last output sometimes.
            if groundtruth_char is not None:
                #one_hot_char = groundtruth_char

                def pick_ground_trouth():
                    """ Return the true value, from the set annotations."""
                    return groundtruth_char
                def pick_last_output():
                    """ Return the last network output (sometimes wrong),
                        to expose the network to false predictions. """
                    return one_hot_char
                def pred():
                    """ Return the network output net_out_prob*100 percent
                        of the time."""
                    net_out_prob = 0.2
                    return tf.greater(rand_val, net_out_prob)

                rand_val = tf.random_uniform(shape=(), minval=0.0, maxval=1.0,
                                             dtype=tf.float32, seed=None,
                                             name='random_uniform')
                one_hot_char = tf.cond(pred(),
                                       pick_ground_trouth,
                                       pick_last_output,
                                       name='truth_or_output_sel')

            #s_i = RNN(s_(i-1), y_(i-1), c_(i-1))
            #Dimensions:   alphabet_size,       32
            #            + listener_output_dim, 40
            #                                   72
            rnn_input = tf.concat(1, [one_hot_char, context_vector],
                                  name='pre_context_rnn_input_concat')
            print('pre_context input size:', tf.Tensor.get_shape(rnn_input))

            pre_context_out, pre_context_states = \
                    self.pre_context_rnn(rnn_input, pre_context_states)


            with tf.variable_scope("attention_context", reuse=reuse):
                ### compute the attention context. ###
                # e_(i,u) = psi(s_i)^T * phi(h_u)
                phi = self.state_net(pre_context_out)
                scalar_energy_lst = []
                #for loop running trough the high level features.
                #assert len(high level features) == U.
                for feat_vec in self.high_lvl_features:
                    psi = self.featr_net(feat_vec)
                    scalar_energy = tf.reduce_sum(psi*phi, reduction_indices=1,
                                                  name='dot_sum')
                    scalar_energy_lst.append(scalar_energy)
                # alpha = softmax(e_(i,u))
                scalar_energy_tensor = tf.convert_to_tensor(scalar_energy_lst)
                #Alpha has the same shape as the scalar_energy_tensor
                alpha = tf.nn.softmax(scalar_energy_tensor)

                #record the scalar_enrgies and alphas for later analysis.
                #TODO: Fix:
                # InvalidArgumentError: All inputs to node MergeSummary/
                # MergeSummary must be from the same frame.
                #Error!!!
                #tf.histogram_summary('scalar_energies', scalar_energy_tensor)
                #tf.histogram_summary('alphas', alpha)

                ### find the context vector. ###
                # c_i = sum(alpha*h_i)
                context_vector = 0*context_vector
                for t in range(0, len(self.high_lvl_features)):
                    #reshaping from (batch_size,) to (batch_size,1) is
                    #needed for broadcasting.
                    current_alpha = tf.reshape(alpha[t, :],
                                               [self.las_model.batch_size,
                                                1])
                    context_vector = (context_vector
                                      + current_alpha*self.high_lvl_features[t])

            if self.type_two is True:
                post_context_out, post_context_states = \
                self.post_context_rnn(context_vector, post_context_states)
                char_net_input = tf.concat(1,
                                           [pre_context_out,
                                            post_context_out],
                                           name='char_net_input_concat')
            else:
                char_net_input = tf.concat(1,
                                           [pre_context_out, context_vector],
                                           name='char_net_input_concat')
            logits = self.char_net(char_net_input)

            # Run the argmax on the character dimension.
            # logits has dimensions [batch_size, num_labels]
            max_pos = tf.argmax(logits, 1, name='choose_max_prob_char')
            one_hot_char = tf.one_hot(max_pos, self.las_model.target_label_no,
                                      axis=1)

            # pack everything up in structures which allow the
            # tensorflow unrolling functions to do their data-type checking.
            attend_and_spell_states = StateTouple(
                RNNStateList(pre_context_states),
                RNNStateList(post_context_states),
                one_hot_char,
                context_vector)
        return logits, attend_and_spell_states


class RNNStateList(list):
    """
    State List class which allows dtype calls necessary because MultiRNNCell,
    stores its output in vanilla python lists, which if used as state variables
    in the Attend and Spell cell cause the tensorflow unrollung function to
    crash, when it checks the data type.    .
    """
    @property
    def dtype(self):
        """Check if the all internal state variables have the same data-type
           if yes return that type. """
        for i in range(1, len(self)):
            if self[i-1].dtype != self[i].dtype:
                raise TypeError("Inconsistent internal state: %s vs %s" %
                                (str(self[i-1].dtype), str(self[i].dtype)))
        return self[0].dtype


class RNN(object):
    """
    Set up the RNN network which computes the decoder state.
    """
    def __init__(self, lstm_dim, name):
        self.name = name
        self.layer_number = 1
        #create the two required LSTM blocks.
        self.blocks = []
        lstm_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        for _ in range(0, self.layer_number):
            self.blocks.append(rnn_cell.LSTMCell(lstm_dim,
                                                 use_peepholes=True,
                                                 state_is_tuple=True,
                                                 initializer=lstm_init))
        self.wrapped_cells = rnn_cell.MultiRNNCell(self.blocks,
                                                   state_is_tuple=True)
        self.reuse = None

    def get_zero_states(self, batch_size, dtype):
        """ Get a list filled with zero states which can be used
            to start up the unrolled LSTM computations."""
        return RNNStateList(self.wrapped_cells.zero_state(batch_size, dtype))

    def __call__(self, single_input, state):
        """
        Computes the RNN outputs for a single input.
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
        #input layer and hidden layers
        for k in range(0, len(self.layers)-1):
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
