'''@file decoder.py
neural network decoder environment'''

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np
from IPython.core.debugger import Tracer; debug_here = Tracer();

class Decoder(object):
    '''Class for the decoding environment for a neural net classifier'''

    def __init__(self, classifier, input_dim, max_length):
        '''
        NnetDecoder constructor, creates the decoding graph

        Args:
            classifier: the classifier that will be used for decoding
            input_dim: the input dimension to the nnnetgraph
        '''

        self.graph = tf.Graph()
        self.max_length = max_length

        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[max_length, input_dim], name='inputs')

            #create the sequence length placeholder
            self.seq_length = tf.placeholder(
                tf.int32, shape=[1], name='seq_length')

            split_inputs = tf.unpack(tf.expand_dims(self.inputs, 1))

            #create the decoding graph
            logits, _, self.saver, _ = classifier(split_inputs,
                                                  self.seq_length,
                                                  is_training=False,
                                                  reuse=False,
                                                  scope='Classifier')

            #convert logits to non sequence for the softmax computation
            logits = seq_convertors.seq2nonseq(logits, self.seq_length)

            #compute the outputs
            self.outputs = tf.nn.softmax(logits)

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

    def __call__(self, inputs):
        '''decode using the neural net

        Args:
            inputs: the inputs to the graph as a NxF numpy array where N is the
                number of frames and F is the input feature dimension

        Returns:
            an NxO numpy array where N is the number of frames and O is the
                neural net output dimension
        '''

        #get the sequence length
        seq_length = [inputs.shape[0]]

        #pad the inputs
        inputs = np.append(
            inputs, np.zeros([self.max_length-inputs.shape[0], inputs.shape[1]])
            , 0)

        #pylint: disable=E1101
        return self.outputs.eval(feed_dict={self.inputs:inputs,
                                            self.seq_length:seq_length})

    def restore(self, filename):
        '''
        load the saved neural net

        Args:
            filename: location where the neural net is saved
        '''

        self.saver.restore(tf.get_default_session(), filename)

class LasDecoder(Decoder):
    def __init__(self, classifier, input_dim, max_length, batch_size):
        '''
        Las Decoder constructor, creates the decoding graph
        The decoder expects a batch size of one utterance for now.

        Args:
            classifier: the classifier that will be used for decoding.
            input_dim: the input dimension to the classifier.
        '''

        self.graph = tf.Graph()
        self.max_length = max_length
        self.input_dim = input_dim
        self.batch_size = batch_size

        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[self.batch_size, max_length, input_dim],
                name='inputs')

            #create the sequence length placeholder
            self.seq_length = tf.placeholder(
                tf.int32, shape=[self.batch_size], name='seq_length')

            #create the decoding graph
            logits, _, self.saver, _ = classifier(self.inputs,
                                                  self.seq_length,
                                                  is_training=False,
                                                  reuse=False,
                                                  scope='Classifier',
                                                  targets=None,
                                                  target_seq_length=None)

            #compute the outputs
            self.outputs = tf.nn.softmax(logits)

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

    def __call__(self, inputs):
        '''decode using the neural net

        Args:
            inputs: the inputs to the graph as a NxF numpy array where N is the
                number of frames and F is the input feature dimension

        Returns:
            an NxO numpy array where N is the number of frames and O is the
                neural net output dimension
        '''

        #get the sequence length
        input_seq_length = [i.shape[0] for i in inputs]

        #pad the inputs with zeros. Their shape will be the same after this.
        inputs = [np.pad(i,
                         ((0, self.max_length-i.shape[0]), (0, 0)),
                         'constant') for i in inputs]

        #pylint: disable=E1101
        return self.outputs.eval(feed_dict={self.inputs:inputs,
                                            self.seq_length:input_seq_length})


class CTCDecoder(Decoder):
    def __init__(self, classifier, input_dim, max_length,
                 max_target_length, batch_size):
        """
        Las Decoder constructor, creates the decoding graph
        The decoder expects a batch size of one utterance for now.

        Args:
            classifier: the classifier that will be used for decoding.
            input_dim: the input dimension to the classifier.
        """

        self.graph = tf.Graph()
        self.max_length = max_length
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.max_target_length = max_target_length

        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[self.batch_size, max_length, input_dim],
                name='inputs')

            #reference labels
            self.targets = tf.placeholder(
                tf.int32, shape=[self.batch_size, max_target_length, 1],
                name='targets')

            idx, vls, shp = self.target_tensor_to_sparse(self.targets)
            sparse_targets = tf.cast(tf.SparseTensor(idx, vls, shp), tf.int32)

            #create the sequence length placeholder
            self.seq_length = tf.placeholder(
                tf.int32, shape=[self.batch_size], name='seq_length')

            #create the decoding graph
            logits, seq_lengths, self.saver, _ = classifier(self.inputs,
                                                            self.seq_length,
                                                            is_training=False,
                                                            reuse=False,
                                                            scope='Classifier',
                                                            targets=None,
                                                            target_seq_length=None)

            #### Evaluating
            
            #predictions = ctc.ctc_beam_search_decoder(logits3d, seq_lengths, beam_width = 100)
            shape = tf.Tensor.get_shape(logits)
            time_mjr_logits = tf.reshape(logits, [int(shape[1]), int(shape[0]), int(shape[2])])
            predictions = ctc.ctc_greedy_decoder(time_mjr_logits, seq_lengths)
            hypothesis = tf.to_int32(predictions[0][0])
            error_rate = tf.reduce_mean(tf.edit_distance(hypothesis,
                                                         sparse_targets,
                                                         normalize=True))

            #compute the outputs
            self.outputs = tf.sparse_to_dense(hypothesis.indices,
                                              hypothesis.shape,
                                              hypothesis.values)
            self.error_rate = error_rate

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

    def target_tensor_to_sparse(self, target_tensor):
        """Make tensorflow SparseTensor from target tensor of shape 
           [numutterances_per_minibatch, max_target_length, 1] with each element
           in the list being a list or array with the values of the target sequence
           (e.g., the integer values of a character map for an ASR target string)
        """
        target_tensor = tf.squeeze(target_tensor)
        #target_list = tf.unpack(target_tensor)
        zero = tf.constant(0, dtype=tf.int32)
        non_zero_mask = tf.not_equal(tf.cast(target_tensor, tf.int32), zero)  
        indices = tf.where(non_zero_mask)   
        vals = tf.boolean_mask(target_tensor, non_zero_mask)
        shape = [self.batch_size, self.max_target_length]
        
        idx = tf.cast(tf.convert_to_tensor(indices), tf.int64)
        vls = tf.cast(tf.convert_to_tensor(vals), tf.int64)
        shp = tf.cast(tf.convert_to_tensor(shape), tf.int64)
        return idx, vls, shp

    def __call__(self, inputs, targets):
        '''decode using the neural net

        Args:
            inputs: the inputs to the graph as a NxF numpy array where N is the
                    number of frames and F is the input feature dimension

        Returns:
            an NxO numpy array where N is the number of frames and O is the
            neural net output dimension
        '''

        #get the sequence length
        input_seq_length = [i.shape[0] for i in inputs]

        #pad the inputs with zeros. Their shape will be the same after this.
        inputs = [np.pad(i,
                         ((0, self.max_length-i.shape[0]), (0, 0)),
                         'constant') for i in inputs]
        inputs = np.array(inputs)

        
        padded_targets = []
        for target in targets:
            padded_target = np.pad(target, (0, self.max_target_length-target.shape[0]),
                                   'constant')
            reshaped_target = np.reshape(padded_target,
                                         [self.max_target_length, 1])
            padded_targets.append(reshaped_target)
        padded_targets = np.array(padded_targets)

        pred = self.outputs.eval(feed_dict={self.inputs:inputs,
                                            self.seq_length:input_seq_length,
                                            self.targets:padded_targets})
        error = self.error_rate.eval(feed_dict={self.inputs:inputs,
                                                self.seq_length:input_seq_length,
                                                self.targets:padded_targets})

        return pred, error
