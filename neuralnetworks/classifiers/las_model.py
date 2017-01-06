'''
This module implements a listen attend and spell classifier.
'''
from __future__ import absolute_import, division, print_function

import sys
import collections
import tensorflow as tf
from tensorflow.python.util import nest

# we are currenly in neuralnetworks, add it to the path.
sys.path.append("neuralnetworks")
from classifiers.classifier import Classifier
from las_elements import Listener
from neuralnetworks.las_elements import AttendAndSpellCell
from IPython.core.debugger import Tracer; debug_here = Tracer();

#interface object containing general model information.
GeneralSettings = collections.namedtuple(
    "GeneralSettings",
    "mel_feature_no, batch_size, target_label_no, dtype")

#interface object containing settings related to the listener.
ListenerSettings = collections.namedtuple(
    "ListenerSettings",
    "lstm_dim, plstm_layer_no, output_dim, out_weights_std, pyramidal")

#interface object containing settings related to the attend and spell cell.
AttendAndSpellSettings = collections.namedtuple(
    "AttendAndSpellSettings",
    "decoder_state_size, feedforward_hidden_units, feedforward_hidden_layers, \
     net_out_prob, type")

class LasModel(Classifier):
    """ A neural end to end network based speech model."""

    def __init__(self, general_settings, listener_settings,
                 attend_and_spell_settings):
        """
        Create a listen attend and Spell model. As described in,
        Chan, Jaitly, Le et al.
        Listen, attend and spell

        Params:
            mel_feature_no: The length of the mel-featrue vectors at each
                            time step.
            batch_size: The number of utterances in each (mini)-batch.
            target_label_no: The number of letters or phonemes in the
                             training data set.
            decoding: Boolean flag indicating if this graph is going to be
                      used for decoding purposes.
        """
        super(LasModel, self).__init__(general_settings.target_label_no)
        self.gen_set = general_settings
        self.lst_set = listener_settings
        self.as_set = attend_and_spell_settings

        self.dtype = tf.float32
        self.mel_feature_no = self.gen_set.mel_feature_no
        self.batch_size = self.gen_set.batch_size
        self.target_label_no = self.gen_set.target_label_no
        self.feat_time = None

        #decoding constants
        self.max_decoding_steps = 100
        #self.max_decoding_steps = 44

        #store the two model parts.
        self.listener = Listener(self.lst_set.lstm_dim, self.lst_set.plstm_layer_no,
                                 self.lst_set.output_dim, self.lst_set.out_weights_std)
        self.attend_and_spell_cell = AttendAndSpellCell(
            self, self.as_set.decoder_state_size,
            self.as_set.feedforward_hidden_units,
            self.as_set.feedforward_hidden_layers,
            self.as_set.net_out_prob,
            self.as_set.type)


    def encode_targets_one_hot(self, targets):
        """
        Transforn the targets into one hot encoded targets.
        Args:
            targets: Tensor of shape [batch_size, max_target_time, 1]
        Returns:
            one_hot_targets: [batch_size, max_target_time, label_no]
        """
        with tf.variable_scope("one_hot_encoding"):
            target_one_hot = tf.one_hot(targets,
                                        self.target_label_no,
                                        axis=2)
            #one hot encoding adds an extra dimension we don't want.
            #squeeze it out.
            target_one_hot = tf.squeeze(target_one_hot, squeeze_dims=[3])
            print("train targets shape: ", tf.Tensor.get_shape(target_one_hot))
            return target_one_hot

    @staticmethod
    def add_input_noise(inputs, stddev=0.65):
        """
        Add noise with a given standart deviation to the inputs
        Args:
            inputs: the noise free input-features.
            stddev: The standart deviation of the noise.
        returns:
            Input features plus noise.
        """
        with tf.variable_scope("input_noise"):
            #add input noise with a standart deviation of stddev.
            inputs = tf.random_normal(tf.shape(inputs), 0.0, stddev) + inputs
        return inputs

    def __call__(self, inputs, seq_length, is_training=False, decoding=False,
                 reuse=True, scope=None, targets=None, target_seq_length=None):
        print('\x1b[01;32m' + "Adding LAS computations:")
        print("    training_graph:", is_training)
        print("    decoding_graph:", decoding)
        print('\x1b[0m')

        if is_training is True:
            inputs = self.add_input_noise(inputs)
            #check if the targets are available for training.
            assert targets is not None

        #inputs = tf.cast(inputs, self.dtype)
        if targets is not None:
            #remove the <sos> token, because training starts at t=1.
            targets_from_t_one = targets[:, 1:, :]
            target_seq_length = target_seq_length-1
            #one hot encode the targets
            target_one_hot = self.encode_targets_one_hot(targets_from_t_one)
        else:
            assert decoding is True, "No targets found. Did you mean to create a decoding graph?"

        input_shape = tf.Tensor.get_shape(inputs)
        print("las input shape:", input_shape)

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            print('adding listen computations to the graph...')
            high_level_features, feature_seq_length \
                = self.listener(inputs, seq_length, reuse)

            if decoding is not True:
                print('adding training attend and spell computations to the graph...')
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
                alphas = None
            else:
                print('adding decoding attend and spell computations to the graph...')
                self.attend_and_spell_cell.set_features(high_level_features,
                                                        feature_seq_length)
                cell_state = self.attend_and_spell_cell.zero_state(
                    self.batch_size, self.dtype)

                with tf.variable_scope('attend_and_spell'):
                    _, _, one_hot_char, _, _ = cell_state
                    logits = tf.expand_dims(one_hot_char, 1)

                    time = tf.constant(0, tf.int32, shape=[])
                    done_mask = tf.cast(tf.zeros(self.batch_size), tf.bool)
                    sequence_length = tf.ones(self.batch_size, tf.int32)
                    alpha = cell_state[-1]
                    #debug_here()
                    self.feat_time = int(tf.Tensor.get_shape(alpha)[1])
                    alphas = tf.expand_dims(alpha, [1])


                    loop_vars = DecodingTouple(logits, alphas, cell_state, time,
                                               done_mask, sequence_length)

                    #set up the shape invariants for the while loop.
                    shape_invariants = loop_vars.get_shape()
                    flat_invariants = nest.flatten(shape_invariants)
                    flat_invariants[0] = tf.TensorShape([self.batch_size,
                                                         None,
                                                         self.target_label_no])
                    flat_invariants[1] = tf.TensorShape([self.batch_size,
                                                         None,
                                                         self.feat_time])
                    shape_invariants = nest.pack_sequence_as(shape_invariants,
                                                             flat_invariants)
                    result = tf.while_loop(
                        self.cond, self.body, loop_vars=[loop_vars],
                        shape_invariants=[shape_invariants])
                    logits, alphas, cell_state, time, _, logits_sequence_length = result[0]

            # The saver can be used to restore the variables in the graph
            # from file later.
            if (is_training is True) or (decoding is True):
                saver = tf.train.Saver()
            else:
                saver = None

        print("Logits tensor shape:", tf.Tensor.get_shape(logits))
        #None is returned as no control ops are defined yet.
        return logits, logits_sequence_length, saver, None, alphas, high_level_features

    def cond(self, loop_vars):
        """ Condition in charge of the attend and spell decoding
            while loop. It checks if all the spellers in the current batch
            are confident of having found an eos token or if a maximum time
            has been exceeded.
        Args:
            loop_vars: The loop variables.
        Returns:
            keep_working, true if the loop should continue.
        """
        _, _, _, time, done_mask, _ = loop_vars

        #the encoding table has the eos token ">" placed at position 0.
        #i.e. ">", "<", ...
        not_done_no = tf.reduce_sum(tf.cast(tf.logical_not(done_mask), tf.int32))
        all_eos = tf.equal(not_done_no, tf.constant(0))
        stop_loop = tf.logical_or(all_eos, tf.greater(time, self.max_decoding_steps))
        keep_working = tf.logical_not(stop_loop)
        #keep_working = tf.Print(keep_working, [keep_working, sequence_length])
        return keep_working


    def get_sequence_lengths(self, time, logits, done_mask, logits_sequence_length):
        """
        Determine the sequence length of the decoded logits based on the
        greedy decoded end of sentence token probability, the current time and
        a done mask, which keeps track of the first appreanche of an end of sentence
        token.

        Args:
            time: The current time step [].
            logits: The logits produced by the las cell in a matrix [batch_size, label_no].
            done_mask: A boolean mask vector of size [batch_size]
            logits_sequence_length: An integer vector with [batch_size] entries.
        Return:
            Updated versions of the logits_sequence_length and mask vectors with unchanged
            sizes.

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

        prev_logits, prev_alphas, cell_state, time, done_mask, logits_sequence_length =  \
            loop_vars
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
        #store the alphas
        alphas = cell_state[-1]
        alphas = tf.expand_dims(alphas, 1)
        alphas = tf.concat(1, [prev_alphas, alphas])
        #pylint: disable=E1101
        alphas.set_shape([self.batch_size, None, self.feat_time])

        out_vars = DecodingTouple(logits, alphas, cell_state, time, done_mask,
                                  logits_sequence_length)
        return [out_vars]

#create a tf style cell state tuple object to derive the actual tuple from.
_DecodingStateTouple = \
    collections.namedtuple(
        "_DecodingStateTouple",
        "logits, alphas, cell_state, time, done_mask, sequence_length")

class DecodingTouple(_DecodingStateTouple):
    """ Tuple used by Attend and spell cells for `state_size`,
     `zero_state`, and output state.
      Stores three elements:
      `(logits, alphas, cell_state, time, done_mask, sequence_length)`, in that order.
      Dimensions are:
            logits:      [batch_size, None, label_no]
            alphas:      [batch_size, None, feature_time]
        cell_state:      A nested list, with the cell variables as outlined in the
                         las cell code.
              time:      A scalar recording the time.
         done_mask:      A boolean mask vector of shape [batch_size] recording
                         where eos tokens have been placed.
        sequence_length: A vector of size [batch_size], recodring the position
                         of the first eos for each batch element.
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
