##@package nnetgraph
# contains the functionality to create neural network graphs and train/test it


# fix some pylint stuff
# fix the pylint import problem.
# pylint: disable=E0401
# remove the python two print statement warning
# pylint: disable=C0325


from abc import ABCMeta, abstractproperty, abstractmethod
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

from neuralnetworks.nnet_las_elements import Listener
from neuralnetworks.nnet_las_elements import AttendAndSpell
from neuralnetworks.nnet_layer import BlstmLayer
from custompython.lazy_decorator import lazy_property
from neuralnetworks.nnet_layer import BlstmSettings
from IPython.core.debugger import Tracer; debug_here = Tracer()


class NnetGraph(object):
    """
    Abstract class difining an interface to be implementd by neural net graphs.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError("Abstract method")

    @property
    def output_dim(self):
        """ Enfore the existance of an output dimension, when
            an NnetGraph object is created.
        """
        raise NotImplementedError("Ouput_dimension must be set.")


class BlstmCtcModel(NnetGraph):
    def __init__(self, name, input_dim, num_hidden_units, max_time_steps,
                 output_dim, input_noise_std):
        self.name = name
        self.input_dim = input_dim
        self.n_hidden = num_hidden_units
        self.max_time_steps = max_time_steps
        self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():
            self.output_dim = output_dim
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
            self.blstm_layer = BlstmLayer(self.output_dim, self.n_hidden, 0.1,
                                          name='BLSTM-Layer')

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


class LasModel(NnetGraph):
    """ A neural end to end network based speech model."""

    def __init__(self, max_time_steps, mel_feature_no, batch_size,
                 target_label_no):
        self.dtype = tf.float32
        self.max_time_steps = max_time_steps
        self.mel_feature_no = mel_feature_no
        self.batch_size = batch_size
        self.target_label_no = target_label_no

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
        print('creating listen functions...')
        self.listen_output_dim = 64
        blstm_settings = BlstmSettings(output_dim=64, lstm_dim=64,
                                       weights_std=0.1, name='blstm')
        plstm_settings = BlstmSettings(self.listen_output_dim,
                                       64, 0.1, 'plstm')
        self.listener = Listener(blstm_settings, plstm_settings, 3,
                                 self.listen_output_dim)

        ###Attend and SPELL
        print('creating attend and spell functions...')
        self.attend_and_spell = AttendAndSpell(self)

    def __call__(self):
        print('adding listen computations to the graph...')
        high_level_features = self.listener(self.input_list,
                                    self.seq_lengths)
        print('adding attend computations to the graph')
        char_dist = self.attend_and_spell(high_level_features)

