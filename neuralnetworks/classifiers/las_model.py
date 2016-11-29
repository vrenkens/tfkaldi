'''
This module implements a listen attend and spell classifier.
'''

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

GeneralSettings = collections.namedtuple(
    "GeneralSettings",
    "mel_feature_no, batch_size, target_label_no, dtype")

ListenerSettings = collections.namedtuple(
    "ListenerSettings",
    "lstm_dim, plstm_layer_no, output_dim, out_weights_std")

AttendAndSpellSettings = collections.namedtuple(
    "AttendAndSpellSettings",
    "decoder_state_size, feedforward_hidden_units, feedforward_hidden_layers")

class LasModel(Classifier):
    """ A neural end to end network based speech model."""

    def __init__(self, general_settings, listener_settings,
                 attend_and_spell_settings, decoding=False):
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
        self.decoding = decoding

        #decoding constants
        self.eos_treshold = 1.8
        #self.eos_treshold = 0.8
        #self.max_decoding_steps = 100
        self.max_decoding_steps = 44

        #store the two model parts.
        self.listener = Listener(self.lst_set.lstm_dim, self.lst_set.plstm_layer_no, 
                                 self.lst_set.output_dim, self.lst_set.out_weights_std)
        self.attend_and_spell_cell = AttendAndSpellCell(
            self, self.as_set.decoder_state_size,
            self.as_set.feedforward_hidden_units,
            self.as_set.feedforward_hidden_layers)


    def __call__(self, inputs, seq_length, is_training=False, reuse=True,
                 scope=None, targets=None, target_seq_length=None):


        print('\x1b[01;32m' + "Adding LAS conputations:")
        print("    training_graph:", is_training)
        print("    decoding_graph:", self.decoding, '\x1b[0m')

        if is_training is True:
            with tf.variable_scope("input_noise"):
                #add input noise with a standart deviation of stddev.
                stddev = 0.65
                inputs = tf.random_normal(tf.shape(inputs), 0.0, stddev) + inputs 


        #inputs = tf.cast(inputs, self.dtype)
        if targets is not None:
            #one hot encode the targets
            with tf.variable_scope("one_hot_encoding"):
                target_one_hot = tf.one_hot(targets,
                                            self.target_label_no,
                                            axis=2)
                #one hot encoding adds an extra dimension we don't want.
                #squeeze it out.
                target_one_hot = tf.squeeze(target_one_hot, squeeze_dims=[3])
                print("train targets shape: ", tf.Tensor.get_shape(target_one_hot))
        else:
            assert self.decoding is True, "Las Training uses the targets."

        input_shape = tf.Tensor.get_shape(inputs)
        print("las input shape:", input_shape)

        if is_training is True:
            assert targets is not None

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            print('adding listen computations to the graph...')
            high_level_features, feature_seq_length \
                = self.listener(inputs, seq_length, reuse)

            if self.decoding is not True:
                print('adding attend and spell computations to the graph...')
                #training mode
                self.attend_and_spell_cell.set_features(high_level_features,
                                                        feature_seq_length)
                zero_state = self.attend_and_spell_cell.zero_state(
                    self.batch_size, self.dtype)
                logits, _ = tf.nn.dynamic_rnn(cell=self.attend_and_spell_cell,
                                              inputs=target_one_hot,
                                              initial_state=zero_state,
                                              time_major=False,
                                              scope='attend_and_spell')
            else:
                print('adding attend and spell computations to the graph...')

                self.attend_and_spell_cell.set_features(high_level_features,
                                                        feature_seq_length)
                cell_state = self.attend_and_spell_cell.zero_state(
                    self.batch_size, self.dtype)

                with tf.variable_scope('attend_and_spell'):
                    _, _, one_hot_char, _ = cell_state
                    logits = tf.expand_dims(one_hot_char, 1)

                    #zero_init = tf.constant_initializer(0)
                    #time = tf.get_variable('time',
                    #                       shape=[],
                    #                       dtype=self.dtype,
                    #                      trainable=False,
                    #                      initializer=zero_init)
                    time = tf.constant(0, self.dtype, shape=[])

                    #turn time from a variable into a tensor.
                    time = tf.identity(time)
                    loop_vars = DecodingTouple(logits, cell_state, time)

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
                    logits, cell_state, time = result[0]

            # The saver can be used to restore the variables in the graph
            # from file later.
            if (is_training is True) or (self.decoding is True):
                saver = tf.train.Saver()
            else:
                saver = None

        print("Logits tensor shape:", tf.Tensor.get_shape(logits))

        #None is returned as no control ops are defined yet.
        return logits, None, saver, None

    def cond(self, loop_vars):
        ''' Condition in charge of the attend and spell decoding
            while loop. It checks if all the spellers in the current batch
            are confident of having found an eos token or if a maximum time
            has been exeeded.'''

        _, cell_state, time = loop_vars
        _, _, one_hot_char, _ = cell_state

        #the encoding table has the eos token ">" placed at position 0.
        #i.e. " ", "<", ">", ...
        eos_prob = one_hot_char[:, 0]
        loop_continue_conditions = tf.logical_and(
            tf.less(eos_prob, self.eos_treshold), #TODO: change to not equal.
            tf.less(time, self.max_decoding_steps))


        loop_continue_counter = tf.reduce_sum(tf.to_int32(
            loop_continue_conditions))
        keep_working = tf.not_equal(loop_continue_counter, 0)
        return keep_working

    def body(self, loop_vars):
        ''' The body of the decoding while loop. Contains a manual enrolling
            of the attend and spell computations.  '''

        prev_logits, cell_state, time = loop_vars
        time = time + 1

        logits, cell_state = \
            self.attend_and_spell_cell(None, cell_state)

        logits = tf.expand_dims(logits, 1)
        logits = tf.concat(1, [prev_logits, logits])
        logits.set_shape([self.batch_size, None, self.target_label_no])

        out_vars = DecodingTouple(logits, cell_state, time)
        return [out_vars]

#create a tf style cell state tuple object to derive the actual tuple from.
_DecodingStateTouple = \
    collections.namedtuple(
        "_DecodingStateTouple",
        "logits, cell_state, time"
        )
class DecodingTouple(_DecodingStateTouple):
    """ Tuple used by Attend and spell cells for `state_size`,
     `zero_state`, and output state.
      Stores three elements:
      `(logits, cell_state, time)`, in that order.
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
