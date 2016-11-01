'''
This module implements a listen attend and spell classifier.
'''

import sys
import tensorflow as tf

# we are currenly in neuralnetworks, add it to the path.
sys.path.append("neuralnetworks")
from classifiers.classifier import Classifier
from las_elements import Listener
from neuralnetworks.las_elements import AttendAndSpellCell
from IPython.core.debugger import Tracer; debug_here = Tracer();


class LasModel(Classifier):
    """ A neural end to end network based speech model."""

    def __init__(self, max_time_steps, mel_feature_no, batch_size,
                 target_label_no):
        super(LasModel, self).__init__(target_label_no)
        self.dtype = tf.float32
        self.max_time_steps = max_time_steps
        self.mel_feature_no = mel_feature_no
        self.batch_size = batch_size
        self.target_label_no = target_label_no
        self.listen_output_dim = 40

        ###LISTENTER
        self.listener = Listener(lstm_dim=56, plstm_layer_no=3,
                                 output_dim=self.listen_output_dim,
                                 out_weights_std=0.1)

        ###Attend and SPELL
        print('creating attend and spell functions...')
        self.attend_and_spell_cell = AttendAndSpellCell(las_model=self)

    def __call__(self, inputs, seq_length, is_training=False, reuse=True,
                 scope=None, targets=None):
        #inputs = tf.cast(inputs, self.dtype)
        targets = tf.cast(targets, self.dtype)

        if is_training is True:
            assert targets is not None

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            print('adding listen computations to the graph...')
            high_level_features = self.listener(inputs,
                                                seq_length)
            print('adding attend computations to the graph...')
            #if is_training is True:
            #training mode
            self.attend_and_spell_cell.set_features(high_level_features)
            zero_state = self.attend_and_spell_cell.zero_state(
                self.batch_size, self.dtype)
            logits, _ = tf.nn.dynamic_rnn(cell=self.attend_and_spell_cell,
                                          inputs=targets,
                                          initial_state=zero_state,
                                          time_major=False)
            #else:
                #TODO: worry about the decoding version of the graph.
            #    logits = None

            # The saver can be used to restore the variables in the graph
            # from file later.
            saver = tf.train.Saver()

        #None is returned as no control ops are defined yet.
        return logits, None, saver, None


