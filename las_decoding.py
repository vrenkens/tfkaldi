# fix some pylint stuff
# fixes the np.random error
# pylint: disable=E1101
# fixes the unrecognized state_is_tuple
# pylint: disable=E1123
# fix the pylint import problem.
# pylint: disable=E0401

import time
import datetime
import pickle
import socket
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from processing.batch_dispenser import TextBatchDispenser
from processing.target_normalizers import aurora4_char_norm
from processing.target_coder import TextEncoder
from processing.feature_reader import FeatureReader
from neuralnetworks.classifiers.las_model import LasModel
from neuralnetworks.decoder import LasDecoder
from IPython.core.debugger import Tracer; debug_here = Tracer()

start_time = time.time()

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
MAX_BATCH_SIZE = 64


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
las_model = LasModel(MEL_FEATURE_NO, MAX_BATCH_SIZE,
                     AURORA_LABELS, decoding=True)

#las_trainer = LasTrainer(las_model, LEARNING_RATE, OMEGA)

max_input_length = np.max([train_dispenser.max_input_length,
                           val_dispenser.max_input_length,
                           test_dispenser.max_input_length])

max_target_length = np.max([train_dispenser.max_target_length,
                            val_dispenser.max_target_length,
                            test_dispenser.max_target_length])


las_decoder = LasDecoder(las_model, MEL_FEATURE_NO, max_time_steps,
                         MAX_BATCH_SIZE)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=las_decoder.graph, config=config):

    las_decoder.restore(
        'saved_models/spchcl23.esat.kuleuven.be-2016-11-10.mdl',
        )

    test_batch = test_dispenser.get_batch()
    inputs = test_batch[0]
    targets = test_batch[1]

    decoded = las_decoder(inputs)

