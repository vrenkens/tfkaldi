import matplotlib.pyplot as plt
import tensorflow as tf
from prepare.batch_dispenser import PhonemeTextDispenser
from prepare.batch_dispenser import UttTextDispenser
from prepare.feature_reader import FeatureReader
from neuralnetworks.nnet_las_model import LasModel
from neuralnetworks.nnet_las_trainer import LasTrainer
from IPython.core.debugger import Tracer; debug_here = Tracer()

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



las_model = LasModel(max_time_steps, MEL_FEATURE_NO, MAX_BATCH_SIZE,
                     AURORA_LABELS)
state = las_model.attend_and_spell_cell.zero_state(las_model.batch_size,
                                                   las_model.dtype)
