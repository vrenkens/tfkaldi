'''module containing the class defininf the blstm ctc model.'''
import sys
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

sys.path.append("neuralnetworks")
from nnet_graph import NnetGraph
from nnet_layer import BlstmLayer
from nnet_layer import BlstmSettings
from custompython.lazy_decorator import lazy_property


class BlstmCtcModel(NnetGraph):
    def __init__(self, name, input_dim, num_hidden_units, max_time_steps,
                 output_dim, input_noise_std):
        super(BlstmCtcModel, self).__init__(output_dim)
        self.name = name
        self.input_dim = input_dim
        self.n_hidden = num_hidden_units
        self.max_time_steps = max_time_steps
        self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():
            self.input_noise_std = input_noise_std

            #TODO: move the computations into a __call()__ !!!

            #Variable wich determines if the graph is for training
            # (if true add noise)
            self.noise_wanted = tf.placeholder(tf.bool, shape=[],
                                               name='add_noise')
            #### Graph input shape=(max_time_steps, batch_size,
            # self.output_dim),
            #    but the first two change.
            self.input_x = tf.placeholder(tf.float32,
                                          shape=(self.max_time_steps,
                                                 None, self.input_dim),
                                          name='melFeatureInput')
            #Prep input data to fit requirements of rnn.bidirectional_rnn
            #Split to get a list of 'n_steps' tensors of shape (batch_size,
            #                                                self.input_dim)
            self.input_list = tf.unpack(self.input_x, num=self.max_time_steps,
                                        axis=0)

            self.seq_lengths = tf.placeholder(tf.int32, shape=None,
                                              name='seqLengths')

            #set up the blstm layer
            blstm_settings = BlstmSettings(self.output_dim, self.n_hidden, 0.1,
                                           name='BLSTM-Layer')
            self.blstm_layer = BlstmLayer(blstm_settings)
            # Make sure all properties are added to the model object upon
            # initialization.
            # pylint does not know how ot deal with the lazy properties.
            # pylint: disable=W0104
            self.input
            self.logits
            self.hypothesis

    @lazy_property
    def input(self):
        """This function adds input noise, when the noise_wanted placeholder
           is set to True."""
        #determine if noise is wanted in this tree.
        def add_noise():
            """Operation used add noise during training"""
            return [tf.random_normal(tf.shape(T), 0.0, self.input_noise_std)
                    + T for T in self.input_list]
        def do_nothing():
            """Operation used to select noise free inputs during validation
            and testing"""
            return self.input_list
        # tf cond applys the first operation if noise_wanted is true and
        # does nothing it the variable is false.
        #local_noise_wanted = tf.Print(self.noise_wanted, [self.noise_wanted],
        #                              message='noise bool val: ')
        blstm_input_list = tf.cond(self.noise_wanted, add_noise,
                                   do_nothing)
        return blstm_input_list

    @lazy_property
    def logits(self):
        """ compute the output layer logits, which in this case is
            done using a linear neuron to combine the results
            computed by the forward and packward lstm passes."""
        # logits3d (max_time_steps, batch_size, n_classes),
        logits = self.blstm_layer(self.input, self.seq_lengths)
        # pack puts the logit list into a big matrix.
        logits3d = tf.pack(logits)
        print("logits 3d shape:", tf.Tensor.get_shape(logits3d))
        return logits3d

    @lazy_property
    def hypothesis(self):
        """
        Decode to compute a the most probable output (hypothesis)
        given the input data.
        """
        predictions = ctc.ctc_greedy_decoder(self.logits,
                                             self.seq_lengths)
        print("predictions", type(predictions))
        print("predictions[0]", type(predictions[0]))
        print("len(predictions[0])", len(predictions[0]))
        print("predictions[0][0]", type(predictions[0][0]))
        hypothesis = tf.to_int32(predictions[0][0])
        return hypothesis