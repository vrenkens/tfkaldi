import sys
import tensorflow as tf

# we are currenly in neuralnetworks, add it to the path.
sys.path.append("neuralnetworks")
from nnet_graph import NnetGraph
from nnet_las_elements import Listener
from nnet_las_elements import AttendAndSpell
from nnet_layer import BlstmSettings


class LasModel(NnetGraph):
    """ A neural end to end network based speech model."""

    def __init__(self, max_time_steps, mel_feature_no, batch_size,
                 target_label_no):
        super(LasModel, self).__init__(target_label_no)
        self.dtype = tf.float32
        self.max_time_steps = max_time_steps
        self.mel_feature_no = mel_feature_no
        self.batch_size = batch_size
        self.target_label_no = target_label_no
        self.listen_output_dim = 64

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

        blstm_settings = BlstmSettings(output_dim=64, lstm_dim=64,
                                       weights_std=0.1, name='blstm')
        plstm_settings = BlstmSettings(self.listen_output_dim,
                                       64, 0.1, 'plstm')
        #TODO: change pLSTM no back to 3!
        self.listener = Listener(blstm_settings, plstm_settings, 1,
                                 self.listen_output_dim)

        ###Attend and SPELL
        print('creating attend and spell functions...')
        self.attend_and_spell = AttendAndSpell(self)

    def __call__(self):
        print('adding listen computations to the graph...')
        high_level_features = self.listener(self.input_list,
                                            self.seq_lengths)
        print('adding attend computations to the graph...')
        char_dist_tensor = self.attend_and_spell(high_level_features)
        return char_dist_tensor


