# fix some pylint stuff
# fixes the np.random error
# pylint: disable=E1101
# fixes the unrecognized state_is_tuple
# pylint: disable=E1123
# fix the pylint import problem.
# pylint: disable=E0401

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
from prepare.batch_dispenser import PhonemeTextDispenser
from prepare.batch_dispenser import UttTextDispenser
from prepare.feature_reader import FeatureReader
from neuralnetworks.nnet_las_model import LasModel
from IPython.core.debugger import Tracer; debug_here = Tracer()


def generate_dispenser(data_path, set_kind, label_no, batch_size, phonemes):
    """ Instatiate a batch dispenser object using the data
        at the spcified path locations"""
    feature_path = data_path + set_kind + "/" + "feats.scp"
    cmvn_path = data_path + set_kind + "/" + "cmvn.scp"
    utt2spk_path = data_path + set_kind + "/" + "utt2spk"
    text_path = data_path + set_kind + "/" + "text"
    featureReader = FeatureReader(feature_path, cmvn_path, utt2spk_path)

    if phonemes is True:
        dispenser = PhonemeTextDispenser(featureReader, batch_size,
                                         text_path, label_no,
                                         max_time_steps)
    else:
        dispenser = UttTextDispenser(featureReader, batch_size,
                                     text_path, label_no,
                                     max_time_steps)
    return dispenser


###Learning Parameters
#LEARNING_RATE = 0.0008
LEARNING_RATE = 0.0008
MOMENTUM = 0.9
#OMEGA = 0.000 #weight regularization term.
OMEGA = 0.001 #weight regularization term.
INPUT_NOISE_STD = 0.6
#LEARNING_RATE = 0.0001       #too low?
#MOMENTUM = 0.6              #play with this.
MAX_N_EPOCHS = 900
OVERFIT_TOL = 0.3


####Network Parameters
n_features = 40
#n_hidden = 164
#n_hidden = 180
n_hidden = 156
#n_hidden = 60

max_time_steps = 2038
AURORA_LABELS = 32
AURORA_PATH = "/esat/spchtemp/scratch/moritz/dataSets/aurora/"
TRAIN = "/train/40fbank"
PHONEMES = False
MAX_BATCH_SIZE = 128
MEL_FEATURE_NO = 40

train_dispenser = generate_dispenser(AURORA_PATH, TRAIN, AURORA_LABELS,
                                     MAX_BATCH_SIZE, PHONEMES)
TEST = "test/40fbank"
val_dispenser = generate_dispenser(AURORA_PATH, TEST, AURORA_LABELS,
                                   128, PHONEMES)

test_dispenser = generate_dispenser(AURORA_PATH, TEST, AURORA_LABELS,
                                    128, PHONEMES)

test_feature_reader = val_dispenser.split_reader(606)
test_dispenser.featureReader = test_feature_reader

BATCH_COUNT = train_dispenser.get_batch_count()
BATCH_COUNT_VAL = val_dispenser.get_batch_count()
BATCH_COUNT_TEST = test_dispenser.get_batch_count()
print(BATCH_COUNT, BATCH_COUNT_VAL, BATCH_COUNT_TEST)
n_classes = AURORA_LABELS


#create the las arcitecture
las_model = LasModel(max_time_steps, MEL_FEATURE_NO, MAX_BATCH_SIZE,
                     AURORA_LABELS)
result = las_model()
