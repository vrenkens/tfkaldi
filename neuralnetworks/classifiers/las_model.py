'''
This module implements a listen attend and spell classifier.
'''

import sys
import collections
import tensorflow as tf

# we are currenly in neuralnetworks, add it to the path.
sys.path.append("neuralnetworks")
from classifiers.classifier import Classifier
from las_elements import Listener
from neuralnetworks.las_elements import AttendAndSpellCell
from IPython.core.debugger import Tracer; debug_here = Tracer();


class LasModel(Classifier):
    """ A neural end to end network based speech model."""

    def __init__(self, mel_feature_no, batch_size,
                 target_label_no, decoding=False):
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
        super(LasModel, self).__init__(target_label_no)
        self.dtype = tf.float32
        self.mel_feature_no = mel_feature_no
        self.batch_size = batch_size
        self.target_label_no = target_label_no
        self.listen_output_dim = 40
        self.decoding = decoding

        #decoding constants
        self.eos_treshold = 0.8
        self.max_decoding_steps = 4000

        ###LISTENTER
        self.listener = Listener(lstm_dim=56, plstm_layer_no=3,
                                 output_dim=self.listen_output_dim,
                                 out_weights_std=0.1)

        ###Attend and SPELL
        self.attend_and_spell_cell = AttendAndSpellCell(las_model=self)

    def __call__(self, inputs, seq_length, is_training=False, reuse=True,
                 scope=None, targets=None, target_seq_length=None):


        print('\x1b[01;32m' + "Adding LAS conputations:")
        print("    training_graph:", is_training)
        print("    decoding_graph:", self.decoding, '\x1b[0m')

        #inputs = tf.cast(inputs, self.dtype)
        if targets is not None:
            targets = tf.cast(targets, self.dtype)
        else:
            assert self.decoding is True, "Las Training uses the targets."

        input_shape = tf.Tensor.get_shape(inputs)
        print("las input shape:", input_shape)

        if is_training is True:
            assert targets is not None

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            print('adding listen computations to the graph...')
            high_level_features = self.listener(inputs,
                                                seq_length)


            if self.decoding is not True:
                print('adding attend and spell computations to the graph...')
                #training mode
                self.attend_and_spell_cell.set_features(high_level_features)
                zero_state = self.attend_and_spell_cell.zero_state(
                    self.batch_size, self.dtype)
                logits, _ = tf.nn.dynamic_rnn(cell=self.attend_and_spell_cell,
                                              inputs=targets,
                                              initial_state=zero_state,
                                              time_major=False,
                                              scope='Listen_and_Spell')
            else:
                print('adding attend and spell computations to the graph...')
                self.attend_and_spell_cell.set_features(high_level_features)
                cell_state = self.attend_and_spell_cell.zero_state(
                    self.batch_size, self.dtype)

                logits = tf.get_variable('logits',
                                         shape=[self.batch_size,
                                                input_shape[1],
                                                self.target_label_no],
                                         dtype=self.dtype,
                                         trainable=False)
                zero_init = tf.constant_initializer(0)
                time = tf.get_variable('time',
                                       shape=[],
                                       dtype=self.dtype,
                                       trainable=False,
                                       initializer=zero_init)
                loop_vars = DecodingTouple(logits, cell_state, time)

                debug_here()
                result = tf.while_loop(self.cond, self.body, loop_vars)

                logits, cell_state, time = result

            # The saver can be used to restore the variables in the graph
            # from file later.
            saver = tf.train.Saver()

        print("Logits tensor shape:", tf.Tensor.get_shape(logits))

        #None is returned as no control ops are defined yet.
        return logits, None, saver, None

    def cond(self, loop_vars):
        ''' Condition in charge of the attend and spell decoding
            while loop. It checks if all the spellers in the current batch
            are confident of having found an eos token or if a maximum time
            has been exeeded.'''

        _, cell_state, time = loop_vars
        _, _, char_dist_vec, _ = cell_state

        #the encoding table has the eos token ">" placed at position 0.
        #i.e. " ", "<", ">", ...
        eos_prob = char_dist_vec[:, 0]
        loop_continue_conditions = tf.logical_and(
            tf.less(eos_prob, self.eos_treshold),
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

        logits = tf.concat(1, prev_logits, logits)
        logits.set_shape([self.batch_size, None, self.target_label_no])

        out_vars = DecodingTouple(logits, cell_state, time)
        return out_vars

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
