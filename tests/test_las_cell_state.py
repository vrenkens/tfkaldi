
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from processing.batch_dispenser import TextBatchDispenser
from processing.target_normalizers import aurora4_char_norm
from processing.target_coder import TextEncoder
from processing.feature_reader import FeatureReader
from neuralnetworks.classifiers.las_model import LasModel
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


###Learning Parameters
#LEARNING_RATE = 0.0008
LEARNING_RATE = 0.0008
LEARNING_RATE_DECAY = 1
MOMENTUM = 0.9
#OMEGA = 0.000 #weight regularization term.
OMEGA = 0.001 #weight regularization term.
#LEARNING_RATE = 0.0001       #too low?
#MOMENTUM = 0.6              #play with this.
MAX_N_EPOCHS = 1
OVERFIT_TOL = 99999

####Network Parameters
n_features = 40


max_time_steps = 2038
AURORA_LABELS = 32
AURORA_PATH = "/esat/spchtemp/scratch/moritz/dataSets/aurora/"
TRAIN = "/train/40fbank"
PHONEMES = False
MAX_BATCH_SIZE = 64
#askoy
if 0:
    UTTERANCES_PER_MINIBATCH = 32 #time vs memory tradeoff.
    DEVICE = '/gpu:0'
#spchcl22
if 1:
    UTTERANCES_PER_MINIBATCH = 32 #time vs memory tradeoff.
    DEVICE = None

MEL_FEATURE_NO = 40




train_dispenser = generate_dispenser(AURORA_PATH, TRAIN, AURORA_LABELS,
                                     MAX_BATCH_SIZE, PHONEMES)
TEST = "test/40fbank"
val_dispenser = generate_dispenser(AURORA_PATH, TEST, AURORA_LABELS,
                                   MAX_BATCH_SIZE, PHONEMES)

test_dispenser = generate_dispenser(AURORA_PATH, TEST, AURORA_LABELS,
                                    MAX_BATCH_SIZE, PHONEMES)

test_feature_reader = val_dispenser.split_reader(606)
test_dispenser.feature_reader = test_feature_reader

BATCH_COUNT = train_dispenser.num_batches
BATCH_COUNT_VAL = val_dispenser.num_batches
BATCH_COUNT_TEST = test_dispenser.num_batches
print(BATCH_COUNT, BATCH_COUNT_VAL, BATCH_COUNT_TEST)
n_classes = AURORA_LABELS

test_batch = test_dispenser.get_batch()
#create the las arcitecture

#mel_feature_no, mini_batch_size, target_label_no, dtype
general_settings = GeneralSettings(n_features, UTTERANCES_PER_MINIBATCH,
                                   AURORA_LABELS, tf.float32)
#lstm_dim, plstm_layer_no, output_dim, out_weights_std
listener_settings = ListenerSettings(256, 3, 256, 0.1)
#decoder_state_size, feedforward_hidden_units, feedforward_hidden_layers
attend_and_spell_settings = AttendAndSpellSettings(512, 512, 3)
las_model = LasModel(general_settings, listener_settings,
                     attend_and_spell_settings)
state = las_model.attend_and_spell_cell.zero_state(las_model.batch_size,
                                                   las_model.dtype)

state_size = las_model.attend_and_spell_cell.state_size

with tf.Session():
    tf.initialize_all_variables().run()
    zero_char = state[2].eval()
    zero_context = state[3].eval()

    print(zero_char.shape)
    print(zero_context.shape)



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


decoded_targets = greedy_search(np.reshape(zero_char, [zero_char.shape[0], 1,
                                                       zero_char.shape[1]]))

print(decoded_targets)

plt.matshow(zero_char)
plt.show()
