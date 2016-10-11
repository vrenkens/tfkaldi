import numpy as np
import tensorflow as tf
from neuralnetworks.nnet_layer import BlstmLayer
from neuralnetworks.nnet_layer import PyramidalBlstmLayer
from neuralnetworks.nnet_layer import BlstmSettings
from neuralnetworks.nnet_layer import FFLayer
from custompython.lazy_decorator import lazy_property

#disable the too few public methods complaint
# pylint: disable=R0903
#disable the print parenthesis warinig coming from python 2 pylint.
# pylint: disable=C0325

class LasModel(object):
    ''' A neural end to end network based speech model.'''

    def __init__(self, max_time_steps, mel_feature_no, batch_size):
        self.dtype = tf.float32
        self.max_time_steps = max_time_steps
        self.mel_feature_no = mel_feature_no
        self.batch_size = batch_size

        #### Graph input shape=(max_time_steps, batch_size, mel_feature_no),
            #    but the first two change.
        self.input_x = tf.placeholder(self.dtype,
                                      shape=(self.max_time_steps,
                                             batch_size, self.mel_feature_no),
                                      name='mel_feature_input')
        #Prep input data to fit requirements of tf.rnn.bidirectional_rnn(')
        #Split to get a list of 'n_steps' tensors of shape
        # (batch_size, self.input_dim)
        self.input_list = tf.unpack(self.input_x, num=self.max_time_steps,
                                    axis=0)

        self.seq_lengths = tf.placeholder(tf.int32, shape=batch_size,
                                          name='seq_lengths')

        ###LISTENTER
        print('setting up the listener')
        self.listen_output_dim = 64
        blstm_settings = BlstmSettings(output_dim=64, lstm_dim=64,
                                       weights_std=0.1, name='blstm')
        plstm_settings = BlstmSettings(self.listen_output_dim,
                                       64, 0.1, 'plstm')
        self.listener = Listener(blstm_settings, plstm_settings, 3,
                                 self.listen_output_dim)
        self.hgh_lvl_fts = self.listener(self.input_list,
                                         self.seq_lengths)

        ###Attend and SPELL
        labels = 33
        print("Setting up the attend and spell part of the graph.")
        self.attend_and_spell = AttendAndSpell(self)


class Listener(object):
    '''
    A set of pyramidal blstms, which compute high level audio features.
    '''
    def __init__(self, blstm_settings, plstm_settings, plstm_layer_no,
                 output_dim):
        ''' initialize the listener.
        '''
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
        ''' Compute the output of the listener function. '''
        #compute the base layer blstm output.
        hidden_values = self.blstm_layer(input_features, sequence_lengths)
        #move on to the plstm ouputs.
        for plstm_layer in self.plstms:
            hidden_values = plstm_layer(hidden_values)
        return hidden_values


class AttendAndSpell(object):
    ''' class implementing alignment establishment and transcription
        or the attend and spell part of the LAS-model.

    Internal Variables:
              features: (H)  the high level features the Listener computed.
         decoder_state: (s_i)
        context_vectors: (c_i) in the paper, found using the
                        attention_context function.
       character_probs: (y) output probability distribution.
    '''
    def __init__(self, las_model):
        self.las_model = las_model
        with tf.variable_scope("attention"):
            self.dec_state_sze = 128
            constant_zero = tf.constant_initializer(0)
            self.decoder_state = tf.get_variable('decoder_state',
                                            shape=[self.dec_state_sze],
                                            dtype=self.las_model.dtype,
                                            initializer=constant_zero,
                                            trainable=False)
            self.state_net = AttentionNetwork(input_dim=self.dec_state_sze,
                                              num_hidden_layers=4,
                                              num_hidden_units=64,
                                              output_dim=64,
                                              name='state_net')
            self.featr_net = AttentionNetwork(input_dim=\
                                              self.las_model.listen_output_dim,
                                              num_hidden_layers=4,
                                              num_hidden_units=64,
                                              output_dim=64,
                                              name='featr_net')
            self.scalar_energy = tf.reduce_sum(
                                   self.state_net(self.decoder_state)
                                  *self.featr_net(self.las_model.hgh_lvl_fts))
            # one dimensional tensor of length U.
            self.alphas = tf.nn.softmax(self.scalar_energy)

            #TODO: unpack things here and compute the context vectors c_i.
            print('high_level_feature_shape')



    def attention_context(self):
        ''' AttentionContext generates a context vector,
         ci encapsulating the information in the acoustic
         signal needed to generate the next character
        '''
        # compute the scalar energy e_(i,u):
        # e_(i,u) = psi(s_i)^T * phi(h_u)

        # compute the attention vector elements alpha (sliding window)
        # alpha = softmax(e_(i,u))

        # find the context vector
        # c_i = sum(alpha*h_i)

    def RNN(self):
        pass

    def Character_distribution(self):
        pass

class AttentionNetwork(object):
    '''A class defining the feedforward MLP networks used to compute the
       scalar energy values required for the attention mechanism.
    '''
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units,
                  output_dim, name):
        self.input_dim = input_dim
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        # Feedforward custom parameters.
        transfername = 'relu'
        dropout = None
        l2_norm = False


        #create the layers
        self.layers = [None]*(self.num_hidden_layers+1)
        #input layer
        self.layers[0] = FFLayer(self.input_dim,
                                 self.num_hidden_units,
                                 1/np.sqrt(self.input_dim),
                                 name + 'layer0',
                                 transfername,
                                 l2_norm,
                                 dropout)
        #hidden layers
        for k in range(1, len(self.layers)-1):
            self.layers[k] = FFLayer(self.num_hidden_units,
                                     self.num_hidden_units,
                                     1/np.sqrt(self.num_hidden_units),
                                     name + 'layer' + str(k),
                                     transfername,
                                     l2_norm,
                                     dropout)
        #output layer
        self.layers[-1] = FFLayer(self.num_hidden_units,
                                  self.output_dim,
                                  0,
                                  name + 'layer' + str(len(self.layers)-1))

    def __call__(self, states_or_features):
        hidden = states_or_features
        for layer in self.layers:
            hidden = layer(hidden, apply_dropout=False)
        return hidden










