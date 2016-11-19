# fix some pylint stuff
# fixes the np.random error
# pylint: disable=E1101
# fixes the unrecognized state_is_tuple
# pylint: disable=E1123
# fix the pylint import problem.
# pylint: disable=E0401

import numpy as np
import pickle
import tensorflow as tf
from processing.batch_dispenser import TextBatchDispenser
from processing.target_normalizers import aurora4_char_norm
from processing.target_coder import TextEncoder
from processing.feature_reader import FeatureReader
from neuralnetworks.classifiers.las_model import LasModel
from neuralnetworks.decoder import LasDecoder
from neuralnetworks.classifiers.las_model import GeneralSettings
from neuralnetworks.classifiers.las_model import ListenerSettings
from neuralnetworks.classifiers.las_model import AttendAndSpellSettings



def generate_dispenser(data_path, set_kind, label_no, batch_size, phonemes):
    """ Instatiate a batch dispenser object using the data
        at the spcified path locations"""
    feature_path = data_path + set_kind + "/" + "feats.scp"
    cmvn_path = data_path + set_kind + "/" + "cmvn.scp"
    utt2spk_path = data_path + set_kind + "/" + "utt2spk"
    text_path = data_path + set_kind + "/" + "text"
    feature_reader = FeatureReader(feature_path, cmvn_path, utt2spk_path,
                                   0, max_time_steps)
    if phonemes is True:
      pass
    #    dispenser = PhonemeTextDispenser(feature_reader, batch_size,
    #                                     text_path, label_no,
    #                                     max_time_steps,
    #                                      one_hot_encoding=True)
    else:
        #Create the las encoder.
        target_coder = TextEncoder(aurora4_char_norm)
        dispenser = TextBatchDispenser(feature_reader,
                                       target_coder,
                                       batch_size,
                                       text_path)
    return dispenser


####Network Parameters
n_features = 40


max_time_steps = 2038
AURORA_LABELS = 32
AURORA_PATH = "/esat/spchtemp/scratch/moritz/dataSets/aurora/"
TRAIN = "/train/40fbank"
PHONEMES = False
MEL_FEATURE_NO = 40

if 0:
    epoch_loss_lst, epoch_loss_lst_val, test_loss, LEARNING_RATE, \
    LEARNING_RATE_DECAY, epoch, UTTERANCES_PER_MINIBATCH, general_settings, \
    listener_settings, attend_and_spell_settings = \
    pickle.load(open("saved_models/spchcl21.esat.kuleuven.be/2016-11-12.pkl", "rb"))
else: 
    MAX_N_EPOCHS = 600
    MAX_BATCH_SIZE = 64
    UTTERANCES_PER_MINIBATCH = 32 #time vs memory tradeoff.
    DEVICE = None
    general_settings = GeneralSettings(n_features, UTTERANCES_PER_MINIBATCH,
                                       AURORA_LABELS, tf.float32)
    #lstm_dim, plstm_layer_no, output_dim, out_weights_std
    listener_settings = ListenerSettings(256, 3, 256, 0.1)
    #decoder_state_size, feedforward_hidden_units, feedforward_hidden_layers
    attend_and_spell_settings = AttendAndSpellSettings(512, 512, 3)


train_dispenser = generate_dispenser(AURORA_PATH, TRAIN, AURORA_LABELS,
                                     UTTERANCES_PER_MINIBATCH, PHONEMES)
TEST = "test/40fbank"
val_dispenser = generate_dispenser(AURORA_PATH, TEST, AURORA_LABELS,
                                   UTTERANCES_PER_MINIBATCH, PHONEMES)

test_dispenser = generate_dispenser(AURORA_PATH, TEST, AURORA_LABELS,
                                    UTTERANCES_PER_MINIBATCH, PHONEMES)

test_feature_reader = val_dispenser.split_reader(606)
test_dispenser.feature_reader = test_feature_reader

BATCH_COUNT = train_dispenser.num_batches
BATCH_COUNT_VAL = val_dispenser.num_batches
BATCH_COUNT_TEST = test_dispenser.num_batches
print(BATCH_COUNT, BATCH_COUNT_VAL, BATCH_COUNT_TEST)
n_classes = AURORA_LABELS

test_batch = test_dispenser.get_batch()
#create the las arcitecture
#load the graph architecutre settings.

las_model = LasModel(general_settings, listener_settings,
                     attend_and_spell_settings, decoding=True)

#las_trainer = LasTrainer(las_model, LEARNING_RATE, OMEGA)

max_input_length = np.max([train_dispenser.max_input_length,
                           val_dispenser.max_input_length,
                           test_dispenser.max_input_length])

max_target_length = np.max([train_dispenser.max_target_length,
                            val_dispenser.max_target_length,
                            test_dispenser.max_target_length])

las_decoder = LasDecoder(las_model, MEL_FEATURE_NO, max_time_steps,
                         UTTERANCES_PER_MINIBATCH)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=las_decoder.graph, config=config):
    las_decoder.restore(
        'saved_models/spchcl21.esat.kuleuven.be/2016-11-16.mdl')
    test_batch = test_dispenser.get_batch()
    #TODO: check input dimension!!! Could be wrong.
    inputs = test_batch[0]
    targets = test_batch[1]
    decoded = las_decoder(inputs)

def greedy_search(network_output):
    """ Extract the largets char probability."""
    utterance_char_batches = []
    for batch in range(0, network_output.shape[0]):
        utterance_chars_nos = []
        for time in range(0, network_output.shape[1]):
            utterance_chars_nos.append(np.argmax(network_output[batch, time, :]))
        utterance_chars = test_dispenser.target_coder.decode(
            utterance_chars_nos)
        utterance_char_batches.append(utterance_chars)
    return np.array(utterance_char_batches)

decoded_targets = greedy_search(decoded)

print(decoded_targets[0])
