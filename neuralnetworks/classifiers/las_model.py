'''
This module implements a listen attend and spell classifier.
'''
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from neuralnetworks.classifiers.classifier import Classifier
from neuralnetworks.las_elements import Listener
from neuralnetworks.las_elements import Speller
from neuralnetworks.beam_search_speller import BeamSearchSpeller
from IPython.core.debugger import Tracer; debug_here = Tracer();


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

        #decoding constants
        self.max_decoding_steps = 100
        #self.max_decoding_steps = 44

        #store the two model parts.
        self.listener = Listener(listener_settings)

        #create a greedy speller.
        #self.speller = Speller(attend_and_spell_settings,
        #                       self.batch_size,
        #                       self.dtype,
        #                       self.target_label_no,
        #                       self.max_decoding_steps)

        #create a beam search speller.
        self.speller = BeamSearchSpeller(attend_and_spell_settings,
                                         self.batch_size,
                                         self.dtype,
                                         self.target_label_no,
                                         self.max_decoding_steps,
                                         beam_width= self.gen_set.beam_width)

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
            targets = targets[:, 1:, :]
            target_seq_length = target_seq_length-1
            #one hot encode the targets
            targets = self.encode_targets_one_hot(targets)
        else:
            assert decoding is True, "No targets found. \
                Did you mean to create a decoding graph?"

        input_shape = tf.Tensor.get_shape(inputs)
        print("las input shape:", input_shape)

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            print('adding listen computations to the graph...')
            high_level_features, feature_seq_length \
                = self.listener(inputs, seq_length, reuse)

            output, output_sequence_length = self.speller(
                high_level_features, feature_seq_length, targets,
                target_seq_length, decoding)

            # The saver can be used to restore the variables in the graph
            # from file later.
            if (is_training is True) or (decoding is True):
                saver = tf.train.Saver()
            else:
                saver = None

        print("output tensor shape:", tf.Tensor.get_shape(output))
        #None is returned as no control ops are defined yet.
        return output, output_sequence_length, saver, None

