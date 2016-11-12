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
from neuralnetworks.classifiers.las_model import GeneralSettings
from neuralnetworks.classifiers.las_model import ListenerSettings
from neuralnetworks.classifiers.las_model import AttendAndSpellSettings

from neuralnetworks.trainer import LasTrainer
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

#las_trainer = LasTrainer(las_model, LEARNING_RATE, OMEGA)

max_input_length = np.max([train_dispenser.max_input_length,
                           val_dispenser.max_input_length,
                           test_dispenser.max_input_length])

max_target_length = np.max([train_dispenser.max_target_length,
                            val_dispenser.max_target_length,
                            test_dispenser.max_target_length])

las_trainer = LasTrainer(
    las_model, n_features, max_input_length, max_target_length,
    LEARNING_RATE, LEARNING_RATE_DECAY, MAX_N_EPOCHS,
    UTTERANCES_PER_MINIBATCH)


print('\x1b[01;32m' + "--- Graph generation done. --- time since start [min]",
     (time.time() - start_time)/60.0, '\x1b[0m')

####Run session
epoch = 0
epoch_loss_lst = []
epoch_loss_lst_val = []

las_trainer.start_visualization('log/' + socket.gethostname() )
#start a tensorflow session
config = tf.ConfigProto()
#pylint does not get the tensorflow object members right.
#pylint: disable=E1101
config.gpu_options.allow_growth = True
with tf.Session(graph=las_trainer.graph, config=config):
    #initialise the trainer
    print("Initializing")
    las_trainer.initialize()
    print("Starting computation.")
    inputs, targets = train_dispenser.get_batch()
    eval_loss = las_trainer.evaluate(inputs, targets)
    epoch_loss_lst.append(eval_loss)
    print("pre training loss:", eval_loss)

    inputs, targets = val_dispenser.get_batch()
    validation_loss = las_trainer.evaluate(inputs, targets)
    epoch_loss_lst_val.append(validation_loss)
    print('validation set pre training loss:', validation_loss)

    continue_training = True
    while continue_training:
        print('Epoch', epoch, '...')
        input_batches = []
        inputs, targets = train_dispenser.get_batch()
        train_loss = las_trainer.update(inputs, targets)
        print('loss:', train_loss)
        epoch = epoch + 1

        if epoch%5 == 0:
            print('\x1b[01;32m'
                  + "-----  validation Step --- time since start [h]   ",
                  (time.time() - start_time)/3600.0,
                  '\x1b[0m')
            epoch_loss_lst.append(train_loss)

            inputs, targets = val_dispenser.get_batch()
            validation_loss = las_trainer.evaluate(inputs, targets)
            print('\x1b[01;32m'
                  + "-----  validation loss: ", validation_loss,
                  '\x1b[0m')
            epoch_loss_lst_val.append(validation_loss)

        # if the training error is lower than the validation error for
        # interval iterations stop..
        #interval = 50
        #if epoch > interval:
        #    print("validation errors",
        #          epoch_loss_lst_val[(epoch - interval):epoch])

        #    val_mean = np.mean(epoch_loss_lst_val[(epoch - interval):epoch])
        #    train_mean = np.mean(epoch_loss_lst[(epoch - interval):epoch])
        #    test_val = val_mean - train_mean - OVERFIT_TOL
        #    print('Overfit condition value:', test_val,
        #          'remaining iterations: ', MAX_N_EPOCHS - epoch)
        #    if (test_val > 0) or (epoch > MAX_N_EPOCHS):
        #        continue_training = False
        #        print("stopping the training.")
        #else:
        #    print("validation losses", epoch_loss_lst_val, epoch)

        if epoch > MAX_N_EPOCHS:
            continue_training = False
            print('\x1b[01;31m', "stopping the training.", '\x1b[0m')

    print('saving the model')
    today = str(datetime.datetime.now()).split(' ')[0]
    filename = "saved_models/" \
               + socket.gethostname() + "/"  \
               + today \
               + ".mdl"
    las_trainer.save_model(filename)
    print("Model saved in file: %s" % filename)

    #run the network on the test data set.
    inputs, targets = test_dispenser.get_batch()
    test_loss = las_trainer.evaluate(inputs, targets)
    print("test loss: ", test_loss)

filename = "saved_models/" \
           + socket.gethostname() \
           + '-' + today \
           + ".pkl"
pickle.dump([epoch_loss_lst, epoch_loss_lst_val, test_loss, LEARNING_RATE,
             MOMENTUM, OMEGA, epoch, general_settings, listener_settings,
             attend_and_spell_settings], open(filename, "wb"))
print("plot values saved at: " + filename)

plt.plot(np.array(epoch_loss_lst))
plt.plot(np.array(epoch_loss_lst_val))
plt.show()

print("--- Program execution done --- time since start [h]",
      (time.time() - start_time)/3600.0)
