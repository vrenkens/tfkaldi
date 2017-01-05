'''This module contains the elements needed to set up the listen attend and
spell network.'''
from __future__ import absolute_import, division, print_function

from copy import copy
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.util import nest

# we are currenly in neuralnetworks, add it to the path.
from neuralnetworks.classifiers.layer import FFLayer
from neuralnetworks.classifiers.layer import PLSTMLayer
from neuralnetworks.classifiers.activation import TfActivation
from neuralnetworks.classifiers.activation import IdentityWrapper
from neuralnetworks.classifiers.seq_convertors import seq2nonseq
from neuralnetworks.classifiers.seq_convertors import nonseq2seq
from IPython.core.debugger import Tracer; debug_here = Tracer();

# disable the too few public methods complaint
# pylint: disable=R0903

#interface object containing general model information.
GeneralSettings = collections.namedtuple(
    "GeneralSettings",
    "mel_feature_no, batch_size, target_label_no, beam_width, dtype")

#interface object containing settings related to the listener.
ListenerSettings = collections.namedtuple(
    "ListenerSettings",
    "lstm_dim, plstm_layer_no, output_dim, out_weights_std, pyramidal")

#interface object containing settings related to the attend and spell cell.
AttendAndSpellSettings = collections.namedtuple(
    "AttendAndSpellSettings",
    "decoder_state_size, feedforward_hidden_units, feedforward_hidden_layers, \
     net_out_prob, type")

class Listener(object):
    """
    A set of pyramidal blstms, which compute high level audio features.
    """
    def __init__(self, listener_settings):
        """ Initialize the listener.

        Arguments:
            A listener settings object containing:
            lstm_dim: The number of LSTM cells per bidirectional layer in the
                      listener.
            plstm_layer_no: The number of plstm compression layers, every
                      layer leads to a time compression by a factor of two.
            output_dim: The output dimension of the listener, for use with CTC.
                        Setting this to NONE produces no output layer.
            out_weights_std: The initial weight initialization standard
                        deviation value.
        """
        self.lstm_dim = int(listener_settings.lstm_dim)
        self.plstm_layer_no = int(listener_settings.plstm_layer_no)
        self.pyramidal = listener_settings.pyramidal
        if listener_settings.output_dim is not None:
            self.output_dim = int(listener_settings.output_dim)
        else:
            self.output_dim = None

        #the listener foundation is a classical bidirectional Long Short
        #term memory layer.
        self.blstm_layer = PLSTMLayer(self.lstm_dim, pyramidal=False)
        #on top of are three pyramidal BLSTM layers.
        self.plstm_layer = PLSTMLayer(self.lstm_dim, pyramidal=self.pyramidal)

        #set an output dimension, when using the Listener together with CTC.
        if self.output_dim != None:
            identity_activation = IdentityWrapper()
            self.ff_layer = FFLayer(self.output_dim, identity_activation,
                                    float(listener_settings.out_weights_std))

    def __call__(self, input_features, sequence_lengths, reuse):
        """ Compute the output of the listener function. """
        # compute the base layer blstm output.
        with tf.variable_scope(type(self).__name__, reuse=reuse):
            hidden_values, sequence_lengths = \
                self.blstm_layer(input_features,
                                 sequence_lengths,
                                 reuse=reuse,
                                 scope="blstm_layer")
            #move on to the plstm outputs.
            for counter in range(self.plstm_layer_no):
                hidden_values, sequence_lengths = \
                    self.plstm_layer(hidden_values,
                                     sequence_lengths,
                                     reuse=reuse,
                                     scope="plstm_layer_" + str(counter))

        if self.output_dim != None:
            with tf.variable_scope('linear_layer', reuse=reuse):
                hidden_shape = tf.Tensor.get_shape(hidden_values)
                non_seq_hidden = seq2nonseq(hidden_values, sequence_lengths)
                non_seq_output_values = self.ff_layer(non_seq_hidden)
                output_values = nonseq2seq(non_seq_output_values,
                                           sequence_lengths,
                                           int(hidden_shape[1]))
        else:
            output_values = hidden_values
        return output_values, sequence_lengths


#create a tf style cell state tuple object to derive the actual tuple from.
_DecodingStateTouple = \
    collections.namedtuple(
        "_DecodingStateTouple",
        "logits, cell_state, time, done_mask, sequence_length")

class DecodingTouple(_DecodingStateTouple):
    """ Tuple used by Attend and spell cells for `state_size`,
     `zero_state`, and output state.
      Stores three elements:
      `(logits, cell_state, time, done_mask, sequence_length)`, in that order.
      Dimensions are:
            logits:      [batch_size, None, label_no]
        cell_state:      A nested list, with the cell variables as outlined in
                         the las cell code.
              time:      A scalar recording the time.
         done_mask:      A boolean mask vector of shape [batch_size] recording
                         where <eos> tokens have been placed.
        sequence_length: A vector of size [batch_size], recording the position
                         of the first <eos> for each batch element.
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

class Speller(object):
    """
    The speller takes high level features and implements an attention based
    transducer to find the desired sequence labeling.
    """

    def __init__(self, as_cell_settings, batch_size, dtype, target_label_no,
                 max_decoding_steps):
        """ Initialize the listener.

        Arguments:
            A speller settings object containing:
            decoder_state_size: The size of the decoder RNN
            feedforward_hidden_units: The number of hidden units in the ff nets.
            feedforward_hidden_layers: The number of hidden layers
                                       for the ff nets.
            net_out_prob: The network output reuse probability during training.
            type: If true a post context RNN is added to the tree.
        """
        self.as_set = as_cell_settings
        self.batch_size = batch_size
        self.dtype = dtype
        self.target_label_no = target_label_no
        self.max_decoding_steps = max_decoding_steps
        self.attend_and_spell_cell = AttendAndSpellCell(
            self, self.as_set.decoder_state_size,
            self.as_set.feedforward_hidden_units,
            self.as_set.feedforward_hidden_layers,
            self.as_set.net_out_prob,
            self.as_set.type)

    def __call__(self, high_level_features, feature_seq_length, target_one_hot,
                 target_seq_length, decoding):
        """
        Arguments:
            high_level_features: The output from the listener
                                 [batch_size, max_input_time, listen_out]
            feature_seq_length: The feature sequence lengths [batch_size]
            target_one_hot: The one hot encoded targets
                                 [batch_size, max_target_time, label_no]
            target_seq_length: Target sequence length vector [batch_size]
            decoding: Flag indicating if a decoding graph must be set up.
        Returns:
            logits: The output logits [batch_size, decoding_time, label_no]
            logits_sequence_length: The logit sequence lengths [batch_size]
            decoded_sequence: Only returned if decoding is True else None
        """

        if decoding is not True:
            print('adding training attend and spell computations ...')
            #training mode
            self.attend_and_spell_cell.set_features(high_level_features,
                                                    feature_seq_length)
            zero_state = self.attend_and_spell_cell.zero_state(
                self.batch_size, self.dtype)
            logits, _ = tf.nn.dynamic_rnn(cell=self.attend_and_spell_cell,
                                          inputs=target_one_hot,
                                          initial_state=zero_state,
                                          sequence_length=target_seq_length,
                                          scope='attend_and_spell')
            logits_sequence_length = target_seq_length
        else:
            print('adding decoding attend and spell computations ...')
            self.attend_and_spell_cell.set_features(high_level_features,
                                                    feature_seq_length)
            cell_state = self.attend_and_spell_cell.zero_state(
                self.batch_size, self.dtype)

            with tf.variable_scope('attend_and_spell'):
                _, _, one_hot_char, _ = cell_state
                logits = tf.expand_dims(one_hot_char, 1)

                time = tf.constant(0, tf.int32, shape=[])
                done_mask = tf.cast(tf.zeros(self.batch_size), tf.bool)
                sequence_length = tf.ones(self.batch_size, tf.int32)
                loop_vars = DecodingTouple(logits, cell_state, time,
                                           done_mask, sequence_length)

                #set up the shape invariants for the while loop.
                shape_invariants = loop_vars.get_shape()
                flat_invariants = nest.flatten(shape_invariants)
                flat_invariants[0] = tf.TensorShape([self.batch_size,
                                                     None,
                                                     self.target_label_no])
                shape_invariants = nest.pack_sequence_as(shape_invariants,
                                                         flat_invariants)
                result = tf.while_loop(
                    self.cond, self.body, loop_vars=[loop_vars],
                    shape_invariants=[shape_invariants])
                logits, cell_state, time, _, logits_sequence_length = result[0]

        return logits, logits_sequence_length

    def cond(self, loop_vars):
        """ Condition in charge of the attend and spell decoding
            while loop. It checks if all the spellers in the current batch
            are confident of having found an eos token or if a maximum time
            has been exeeded.
        Args:
            loop_vars: The loop variables.
        Returns:
            keep_working, true if the loop should continue.
        """
        _, _, time, done_mask, _ = loop_vars

        #the encoding table has the eos token ">" placed at position 0.
        #i.e. ">", "<", ...
        not_done_no = tf.reduce_sum(tf.cast(tf.logical_not(done_mask),
                                            tf.int32))
        all_eos = tf.equal(not_done_no, tf.constant(0))
        stop_loop = tf.logical_or(all_eos, tf.greater(time,
                                                      self.max_decoding_steps))
        keep_working = tf.logical_not(stop_loop)
        #keep_working = tf.Print(keep_working, [keep_working, sequence_length])
        return keep_working


    def get_sequence_lengths(self, time, logits, done_mask,
                             logits_sequence_length):
        """
        Determine the sequence length of the decoded logits based on the
        greedy decoded end of sentence token probability, the current time and
        a done mask, which keeps track of the first appreanche of an end of
        sentence token.

        Args:
            time: The current time step [].
            logits: The logits produced by the las cell in a matrix
                    [batch_size, label_no].
            done_mask: A boolean mask vector of size [batch_size]
            logits_sequence_length: An integer vector with [batch_size] entries.
        Return:
            Updated versions of the logits_sequence_length and mask
            vectors with unchanged sizes.
        """
        with tf.variable_scope("get_sequence_lengths"):

            max_vals = tf.argmax(logits, 1)
            mask = tf.equal(max_vals, tf.constant(0, tf.int64))
            #current_mask = tf.logical_and(mask, tf.logical_not(done_mask))

            time_vec = tf.ones(self.batch_size, tf.int32)*(time+1)
            logits_sequence_length = tf.select(done_mask,
                                               logits_sequence_length,
                                               time_vec)
            done_mask = tf.logical_or(mask, done_mask)
        return done_mask, logits_sequence_length

    def body(self, loop_vars):
        ''' The body of the decoding while loop. Contains a manual enrolling
            of the attend and spell computations.
        Args:
            The loop variables from the previous iteration.
        Returns:
            The loop variables as computed during the current iteration.
        '''

        prev_logits, cell_state, time, done_mask, \
            logits_sequence_length = loop_vars
        time = time + 1

        logits, cell_state = \
            self.attend_and_spell_cell(None, cell_state)

        #update the sequence lengths.
        done_mask, logits_sequence_length = self.get_sequence_lengths(
            time, logits, done_mask, logits_sequence_length)

        #store the logits.
        logits = tf.expand_dims(logits, 1)
        logits = tf.concat(1, [prev_logits, logits])
        #pylint: disable=E1101
        logits.set_shape([self.batch_size, None, self.target_label_no])

        out_vars = DecodingTouple(logits, cell_state, time,
                                  done_mask, logits_sequence_length)
        return [out_vars]


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

    def to_tensor(self):
        """ This op turns the touple into a tensor.
        Returns:
            state_tensor: A tensor with the list contents concatenated along one dimension.
            element_lengths: A tensor with the length of each element in the state_tensor.
        """
        with tf.variable_scope("StateTouple_to_StateTensor"):
            flat_self = nest.flatten(self)
            
            squeezed_tensors = []
            for i in range(0, len(flat_self)):
                # state touple elements have the shape [Dimension(1), Dimension(?)].
                # the ? depends on the network parameters. The zeroth dimension is not 
                # interesting.
                squeezed_element = tf.squeeze(flat_self[i], [0])
                squeezed_tensors.append(squeezed_element)
            state_tensor = tf.concat(0, squeezed_tensors)
            return state_tensor

    def get_element_lengths(self):
        """
        Get the length of individual state elements as found in a concatenated
        state Tensor.
        Returns:
            The element lengths of the single touple entries. 
        """
        flat_self = nest.flatten(self)            
        element_lengths = []
        for i in range(0, len(flat_self)):
            # state touple elements have the shape [Dimension(1), Dimension(?)].
            # the ? depends on the network parameters. The zeroth dimension is not 
            # interesting.
            squeezed_element = tf.squeeze(flat_self[i], [0])
            element_lengths.append(int(tf.Tensor.get_shape(squeezed_element)[0]))
        return element_lengths

    def to_list(self, state_tensor, element_lengths):
        """
        Take a state tensor and pack it into a list with the same
        structure of self.
        WARNING: The code assumes that the state_tensor input does indeed
                 fit into the state list and that tensor sizes in element_lengths
                 are correct.
                 This assumption is not checked, if it's violated strange
                 things will happen.
        Args:
            state_tensor: A concatinated attend and spell state tensor.
            element_lengths: The element lengths of the state tensor.
        Returns:
            A repacked AttendAndSpellStateTouple container object.
        """
        
        with tf.variable_scope("StateTensor_to_StateTouple"):
            flat_self = []
            start = 0
            for length in element_lengths:
                stop = start + length
                list_element = tf.gather(state_tensor, tf.range(start, stop))
                list_element = tf.reshape(list_element, [1, length])
                flat_self.append(list_element)
                start = stop
        return nest.pack_sequence_as(self, flat_self)


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
                 feedforward_hidden_units=56, feedforward_hidden_layers=3,
                 net_out_prob=0.2, type_two=False):
        self.feedforward_hidden_units = int(feedforward_hidden_units)
        self.feedforward_hidden_layers = int(feedforward_hidden_layers)
        self.net_out_prob = float(net_out_prob)
        #the decoder state size must be equal to the RNN size.
        self.dec_state_size = int(decoder_state_size)
        self.high_lvl_features = None
        self.high_lvl_feature_dim = None
        self.psi = None

        self.las_model = las_model

        #Determines whether the post_context_rnn will be used.
        self.type_two = type_two

        #--------------------Create network functions-------------------------#
        # Feed-forward layer custom parameters. Vincent knows more about these.
        activation = None
        activation = TfActivation(activation, tf.nn.relu)

        state_net_dimension = FFNetDimension(self.feedforward_hidden_units,
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
            output_dim=self.las_model.target_label_no,
            num_hidden_units=self.feedforward_hidden_units,
            num_hidden_layers=self.feedforward_hidden_layers)

        self.char_net = FeedForwardNetwork(char_net_dimension,
                                           activation,
                                           name='char_net')

    def set_features(self, high_lvl_features, feature_seq_lengths):
        ''' Set the features when available, storing the features in the
            object makes the cell call simpler. Additionally this function
            evaluates the state net and stores the result moving this
            computation out of the loop for efficiency. The computed
            data is stored in the cell object for future reference.
        Args:
            high_lvl_featrues: The output computed by the listener. [batch_size,
                                compresses_max_time, feature_dim]
            feature_seq_lengths: The feature sequence lengths. [batch_size]'''

        self.high_lvl_features = high_lvl_features
        feature_shape = tf.Tensor.get_shape(high_lvl_features)

        with tf.variable_scope("compute_psi"):
            print("     Feature dimension:", feature_shape)
            self.high_lvl_feature_dim = feature_shape[2]
            feature_shape = tf.Tensor.get_shape(high_lvl_features)
            non_seq_features = seq2nonseq(high_lvl_features, feature_seq_lengths)
            non_seq_psi = self.featr_net(non_seq_features)
            self.psi = nonseq2seq(non_seq_psi, feature_seq_lengths,
                                  int(feature_shape[1]))
            print("  Psi tensor dimension:", tf.Tensor.get_shape(self.psi))

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
        Args:
            batch_size: The size of the mini-batches, which are going to be fed into
                        this instantiation of this classifier.
            dtype: The data type used for the model.
        Returns:
            A StateTouple object filled with the state variables.
        """
        #The batch_size has to be fixed in order to be able to correctly
        #return the state_sizes, should self.state_size() be called before
        #the zero states are created.
        assert batch_size == self.las_model.batch_size
        assert dtype == self.las_model.dtype

        zero_state_scope = type(self).__name__+ "_zero_state"
        with tf.variable_scope(zero_state_scope):
            #----------------------Create Zero state tensors------------------#
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
                                        self.high_lvl_feature_dim])
            context_vector = tf.constant(zero_context_np, dtype)
            #context_vector = tf.identity(context_vector)
            print("  context_vector shape:",
                  tf.Tensor.get_shape(context_vector))

        return StateTouple(pre_context_states, post_context_states,
                           one_hot_char, context_vector)

    def select_out_or_target(self, groundtruth_char, one_hot_char):
        """ Select the last system output value, or the ground-truth.
            The probability of picking the ground truth value is
            given by self.net_out_prob value. """

        def pick_ground_truth():
            """ Return the true value, from the set annotations."""
            return groundtruth_char
        def pick_last_output():
            """ Return the last network output (sometimes wrong),
                to expose the network to false predictions. """
            return one_hot_char
        def pred():
            """ Return the network output net_out_prob*100 percent
                of the time."""
            return tf.greater(rand_val, self.net_out_prob)

        rand_val = tf.random_uniform(shape=(), minval=0.0, maxval=1.0,
                                     dtype=tf.float32, seed=None,
                                     name='random_uniform')
        one_hot_char = tf.cond(pred(),
                               pick_ground_truth,
                               pick_last_output,
                               name='truth_or_output_sel')
        return one_hot_char

    def greedy_decoding(self, logits):
        """ Greedy decoding, select the output character, with the
            largest corresponding logit value.
        Args:
            logits: the las output logits for the current batch.
                    dimensions [batch_size, num_labels].
        Returns:
            one_hot_char: One hot encoded selected characters.
                          shape [batch_size, num_labels].
        """
        with tf.variable_scope("greedy_decoding"):
            max_pos = tf.argmax(logits, 1, name='choose_max_prob_char')
            one_hot_char = tf.one_hot(max_pos, self.las_model.target_label_no,
                                      axis=1)
        return one_hot_char


    def attention_context(self, pre_context_out):
        """ Compute the attention context based on the high lefel features
            and the pre context state.
        Args:
            pre_context_out: The output of the state RNN, s_i in the las
                             paper.
        Returns:
            context_vector: The context vector a linear combination of the
                            high level features computed by the listener.

        """
        with tf.variable_scope("attention_context"):
            ### compute the attention context. ###
            # e_(i,u) = phi(s_i)^T * psi(h_u)
            phi = self.state_net(pre_context_out)
            # phi_3d shape: [batch_size, state_size, 1]
            phi_3d = tf.expand_dims(phi, 2)
            # [batch_size, time, state_size] * [batch_size, state_size, 1]
            # = [batch_size, time, 1]
            energy_3d = tf.batch_matmul(self.psi, phi_3d,
                                        name='scalar_energy_matmul')
            scalar_energy = tf.squeeze(energy_3d, squeeze_dims=[2])
            alpha = tf.nn.softmax(scalar_energy)

            ### find the context vector. ###
                # c_i = sum(alpha*h_i)
            # alpha_3d has shape: [batch_size, 1 , time].
            alpha_3d = tf.expand_dims(alpha, 1)
            # [batch_size, 1 , time] * [batch_size, time, state_dim]
            # = [batch_size, 1, state_dim]
            context_vector_3d = tf.batch_matmul(alpha_3d,
                                                self.high_lvl_features,
                                                name='context_matmul')
            context_vector = tf.squeeze(context_vector_3d, squeeze_dims=[1])
        return context_vector


    def __call__(self, cell_input, state, scope=None, reuse=False):
        """
        Do the computations for a single unrolling of the attend and
        spell network.
        Arguments:
            cell_input: During training make sure the cell_input contains
                        valid ground-truth values.
                        During decoding the cell_input may be none.
            state: The attend and spell cell state, must be a cell state
                   Tuple object containing the variables
                   pre_context_states, post_context_states,
                   one_hot_char, context_vector, in that order.
            scope: a scope name for the cell.
            reuse: The reuse flag, set to true to reuse variables previously
                   defined.
        Returns:
            An attend and spell state tuple object with updated values.
        """
        as_cell_call_scope = scope or (type(self).__name__+ "_call")
        with tf.variable_scope(as_cell_call_scope, reuse=reuse):
            groundtruth_char = cell_input
            # StateTouple extends a collection, pylint doesn't get it.
            # pylint: disable = E0633
            pre_context_states, post_context_states, one_hot_char, \
                context_vector = state

            if self.psi is None:
                raise AttributeError("Features must be set.")

            #make sure no targets are present during decoding.
            #if decoding_tree is True:
            #    assert groundtruth_char is None, \
            #        "Targets cannot be set during decoding."

            #Pick the last output sometimes.
            if groundtruth_char is not None:
                #one_hot_char = groundtruth_char
                one_hot_char = self.select_out_or_target(groundtruth_char,
                                                         one_hot_char)

            ### Compute the attend and spell state ###
            #s_i = RNN(s_(i-1), y_(i-1), c_(i-1))
            rnn_input = tf.concat(1, [one_hot_char, context_vector],
                                  name='pre_context_rnn_input_concat')
            print('pre_context input size:', tf.Tensor.get_shape(rnn_input))

            pre_context_out, pre_context_states = \
                    self.pre_context_rnn(rnn_input, pre_context_states)

            ### compute the attention context. ###
            context_vector = self.attention_context(pre_context_out)

            #add the post context rnn layer for type two cells.
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
            one_hot_char = self.greedy_decoding(logits)

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
    State List class which allows dtype calls. Necessary because MultiRNNCell,
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
            self.blocks.append(rnn_cell.LSTMCell(int(lstm_dim),
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
        Args:
            single_input: The input, for which the output given the
                          state should be computed.
            state: The las state
        Returns:
            A touple output, new state.
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
    def __init__(self, output_dim, num_hidden_units,
                 num_hidden_layers):
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
        #linar output layer, no activation required.
        identity_activation = IdentityWrapper()
        self.layers[-1] = FFLayer(dimension.output_dim, identity_activation)

    def __call__(self, states_or_features):
        """
        Evaluate this feedforward net given the current input.
        Args:
            states_or_features: The state or feature vector this network
                                should be evaluated on.
        Returns:
            The network output.
        """
        hidden = states_or_features
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, scope=(self.name + '/' + str(i)),
                           reuse=(self.reuse))
        #set reuse to true after the variables have been created in the first
        #call.
        if self.reuse is None:
            self.reuse = True
        return hidden
