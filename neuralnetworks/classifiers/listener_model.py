#A ctc model which will be used to verify the input code.
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
from IPython.core.debugger import Tracer; debug_here = Tracer();


GeneralSettings = collections.namedtuple(
    "GeneralSettings",
    "mel_feature_no, batch_size, target_label_no, dtype")

ListenerSettings = collections.namedtuple(
    "ListenerSettings",
    "lstm_dim, plstm_layer_no, output_dim, out_weights_std")

class ListenerModel(Classifier):
    """ A neural end to end network based speech model."""

    def __init__(self, general_settings, listener_settings, decoding=False):
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
        super(ListenerModel, self).__init__(general_settings.target_label_no)
        self.gen_set = general_settings
        self.lst_set = listener_settings

        self.dtype = tf.float32
        self.mel_feature_no = self.gen_set.mel_feature_no
        self.batch_size = self.gen_set.batch_size
        self.target_label_no = self.gen_set.target_label_no
        self.decoding = decoding

        #decoding constants
        self.eos_treshold = 0.8
        self.max_decoding_steps = 4000

        #store the two model parts.
        self.listener = Listener(self.lst_set.lstm_dim, self.lst_set.plstm_layer_no, 
                                 self.lst_set.output_dim, self.lst_set.out_weights_std)

    def __call__(self, inputs, seq_length, is_training=False, reuse=True,
                 scope=None, targets=None, target_seq_length=None):

        print('\x1b[01;32m' + "Adding LAS computations:")
        print("    training_graph:", is_training)
        print("    decoding_graph:", self.decoding)
        print('\x1b[0m')

        input_shape = tf.Tensor.get_shape(inputs)
        print("las input shape:", input_shape)

        if is_training is True:
            with tf.variable_scope("input_noise"):
                #add input noise with a standart deviation of stddev.
                stddev = 0.65
                inputs = tf.random_normal(tf.shape(inputs), 0.0, stddev) + inputs  

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            print('adding listen computations to the graph...')
            high_level_features, seq_length = self.listener(inputs,
                                                            seq_length,
                                                            reuse)
        logits = high_level_features

        if (is_training is True) or (self.decoding is True):
            saver = tf.train.Saver()
        else:
            saver = None

        print("Logits tensor shape:", tf.Tensor.get_shape(logits))
        #None is returned as no control ops are defined yet.
        return logits, seq_length, saver, None

 