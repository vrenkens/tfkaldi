import numpy as np
import tensorflow as tf
from copy import copy
from neuralnetworks.nnet_layer import BlstmSettings
from neuralnetworks.nnet_layer import BlstmLayer
from neuralnetworks.nnet_layer import PyramidalBlstmLayer
from neuralnetworks.nnet_layer import FFSettings
from neuralnetworks.nnet_layer import FFLayer
from custompython.lazy_decorator import lazy_property

#disable the too few public methods complaint
# pylint: disable=R0903
#disable the print parenthesis warinig coming from python 2 pylint.
# pylint: disable=C0325

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
    ''' Class implementing alignment establishment and transcription
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
            self.feedforward_hidden_units = 42
            self.feedforward_hidden_layers = 4
            self.dec_state_size = 42

            # Feedforward custom parameters.
            transfername = 'relu'
            dropout = None
            l2_norm = False


            state_net_settings = AttenionNetSettings(self.dec_state_size,
                                                self.feedforward_hidden_units,
                                                self.feedforward_hidden_units,
                                                self.feedforward_hidden_layers)

            state_layer_settings = FFSettings(input_dim=None, #assigned later
                                              output_dim=None,
                                              weights_std=None,
                                              name='state_net',
                                              transfername=transfername,
                                              l2_norm=l2_norm,
                                              dropout=dropout)

            self.state_net = AttentionNetwork(state_net_settings,
                                              state_layer_settings)

            # copy the state net any layer settings
            # => all properties, which are not explicitly changed
            # stay the same.
            featr_net_settings = copy(state_net_settings)
            featr_net_settings.input_dim = self.las_model.listen_output_dim

            featr_layer_settings = copy(state_layer_settings)
            featr_layer_settings.name = 'feature_net'

            self.featr_net = AttentionNetwork(featr_net_settings,
                                              featr_layer_settings)


            #self.decoder_states = []
            #for idx,h in enumerate(self.las_model.hgh_lvl_fts):
            #    tf.reduce_sum(self.featr_net(h)
            #                  *self.state_net(decoder_states[idx])

            # one dimensional tensor of length U.
            #self.alphas = tf.nn.softmax(self.scalar_energy)

            #TODO: unpack things here and compute the context vectors c_i.
            #print('high_level_feature_shape')



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
    def Character_distribution(self):
        pass


class RNN(object):
    '''
    Set up the RNN network which computes the decoder state.
    This function takes
    '''
    def __init__(self, layer_number, name):
      #create the two required LSTM blocks.
      self.blocks = []
      for i in range(0,layer_number):
        with tf.variable_scope(name + 'block' + str(i)):
          self.blocks.append(rnn_cell.LSTMCell(settings.lstm_dim,
                                             use_peepholes=True,
                                             state_is_tuple=True))

    def _call__(self):
      pass
      #TODO: Implement.



class AttenionNetSettings(object):
    """docstring for AttenionNetSettings"""
    def __init__(self, input_dim, output_dim, num_hidden_units,
                 num_hidden_layers):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers


class AttentionNetwork(object):
    '''A class defining the feedforward MLP networks used to compute the
       scalar energy values required for the attention mechanism.
    '''
    def __init__(self, attention_net_settings, fflayer_settings):
        #create shorter namespaces
        ats = attention_net_settings
        ffs = fflayer_settings
        #store the settings
        self.attention_net_settings = ats
        self.fflayer_settings = ffs

        #create the layers
        self.layers = [None]*(ats.num_hidden_layers+1)
        #input layer
        ffs.input_dim = ats.input_dim
        ffs.name = ffs.name + '_layer0'
        ffs.output_dim = ats.num_hidden_layers
        ffs.weights_std = 1/np.sqrt(ats.input_dim)
        self.layers[0] = FFLayer(ffs)
        #hidden layers
        for k in range(1, len(self.layers)-1):
            ffs.input_dim = ats.num_hidden_units
            ffs.weights_std = 1/np.sqrt(ats.num_hidden_units)
            ffs.name = ffs.name + '_layer' + str(k)
            self.layers[k] = FFLayer(ffs)
        #output layer
        ffs.output_dim = ats.output_dim
        ffs.name = ffs.name \
                                + '_layer' + str(len(self.layers)-1)
        self.layers[-1] = FFLayer(ffs)

    def __call__(self, states_or_features):
        hidden = states_or_features
        for layer in self.layers:
            hidden = layer(hidden, apply_dropout=False)
        return hidden



