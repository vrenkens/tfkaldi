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
from processing.batch_dispenser import PhonemeBatchDispenser
from processing.target_normalizers import timit_phone_norm
from processing.target_coder import PhonemeEncoder
from processing.feature_reader import FeatureReader
from neuralnetworks.classifiers.listener_model import ListenerModel
from neuralnetworks.classifiers.las_model import GeneralSettings
from neuralnetworks.classifiers.las_model import ListenerSettings
from neuralnetworks.trainer import CTCTrainer

from neuralnetworks.trainer import LasTrainer
from IPython.core.debugger import Tracer; debug_here = Tracer()

def set_up_dispensers(max_batch_size):
    """load training, validation and testing data.""" 

    #training data --------------------------------------------------------
    timit_path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/"
    feature_path = timit_path  + "fbank/" + "raw_fbank_train.1.scp"
    cmvn_path = None
    utt2spk_path = timit_path + "data/train/utt2spk"
    feature_reader = FeatureReader(feature_path, cmvn_path, utt2spk_path,
                                   0, MAX_TIME_STEPS_TIMIT)
    train_phone_path = timit_path + "data/train/train39.text"
    target_coder = PhonemeEncoder(timit_phone_norm) 
    target_path = train_phone_path
    #self, feature_reader, target_coder, size, target_path

    traindispenser = PhonemeBatchDispenser(feature_reader, target_coder, max_batch_size, 
                                           target_path)

    #validation data ------------------------------------------------------
    feature_path = timit_path  + "fbank/" + "raw_fbank_dev.1.scp"
    cmvn_path = None
    utt2spk_path = timit_path + "data/dev/utt2spk"
    feature_reader = FeatureReader(feature_path, cmvn_path, utt2spk_path,
                                   0, MAX_TIME_STEPS_TIMIT)
    train_phone_path = timit_path + "data/dev/dev39.text"
    target_coder = PhonemeEncoder(timit_phone_norm) 
    target_path = train_phone_path
    #self, feature_reader, target_coder, size, target_path

    valdispenser = PhonemeBatchDispenser(feature_reader, target_coder, max_batch_size, 
                                         target_path)

    #######test data-------------------------------------------------------
    feature_path = timit_path  + "fbank/" + "raw_fbank_test.1.scp"
    cmvn_path = None
    utt2spk_path = timit_path + "data/test/utt2spk"
    feature_reader = FeatureReader(feature_path, cmvn_path, utt2spk_path,
                                   0, MAX_TIME_STEPS_TIMIT)
    train_phone_path = timit_path + "data/test/test39.text"
    target_coder = PhonemeEncoder(timit_phone_norm) 
    target_path = train_phone_path
    #self, feature_reader, target_coder, size, target_path

    testdispenser = PhonemeBatchDispenser(feature_reader, target_coder, max_batch_size, 
                                          target_path)

    return traindispenser, valdispenser, testdispenser

start_time = time.time()

###Learning Parameters
#LEARNING_RATE = 0.0008
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.98

OVERFIT_TOL = 99999

####Network Parameters
n_features = 40


TIMIT_LABELS = 39
CTC_TIMIT_LABELS = TIMIT_LABELS + 1 #add one for ctc

#askoy
if 1:
    MAX_N_EPOCHS = 600
    MAX_BATCH_SIZE = 64
    UTTERANCES_PER_MINIBATCH = 32 #time vs memory tradeoff.
    #mel_feature_no, mini_batch_size, target_label_no, dtype
    general_settings = GeneralSettings(n_features, UTTERANCES_PER_MINIBATCH,
                                       CTC_TIMIT_LABELS, tf.float32)
    #lstm_dim, plstm_layer_no, output_dim, out_weights_std
    listener_settings = ListenerSettings(156, 0, 56, 0.1)


MAX_TIME_STEPS_TIMIT = 777
MEL_FEATURE_NO = 40

train_dispenser, val_dispenser, test_dispenser = set_up_dispensers(MAX_BATCH_SIZE)

BATCH_COUNT = train_dispenser.num_batches
BATCH_COUNT_VAL = val_dispenser.num_batches
BATCH_COUNT_TEST = test_dispenser.num_batches
print(BATCH_COUNT, BATCH_COUNT_VAL, BATCH_COUNT_TEST)
n_classes = CTC_TIMIT_LABELS

test_batch = test_dispenser.get_batch()
#create the las arcitecture

max_input_length = np.max([train_dispenser.max_input_length,
                           val_dispenser.max_input_length,
                           test_dispenser.max_input_length])

max_target_length = np.max([train_dispenser.max_target_length,
                            val_dispenser.max_target_length,
                            test_dispenser.max_target_length])
listener_model = ListenerModel(general_settings, listener_settings)

ctc_trainer = CTCTrainer(listener_model, MEL_FEATURE_NO, MAX_TIME_STEPS_TIMIT,
                         max_target_length, LEARNING_RATE, LEARNING_RATE_DECAY,
                         MAX_N_EPOCHS, UTTERANCES_PER_MINIBATCH)


print('\x1b[01;32m' + "--- Graph generation done. --- time since start [min]",
      (time.time() - start_time)/60.0, '\x1b[0m')

####Run session
epoch = 0
epoch_loss_lst = []
epoch_loss_lst_val = []

ctc_trainer.start_visualization('log/' + socket.gethostname())
#start a tensorflow session
config = tf.ConfigProto()
#pylint does not get the tensorflow object members right.
#pylint: disable=E1101
config.gpu_options.allow_growth = True
with tf.Session(graph=ctc_trainer.graph, config=config):
    #initialise the trainer
    print("Initializing")
    ctc_trainer.initialize()
    print("Starting computation.")
    inputs, targets = train_dispenser.get_batch()
    eval_loss = ctc_trainer.evaluate(inputs, targets)
    epoch_loss_lst.append(eval_loss)
    print("pre training loss:", eval_loss)

    inputs, targets = val_dispenser.get_batch()
    validation_loss = ctc_trainer.evaluate(inputs, targets)
    epoch_loss_lst_val.append(validation_loss)
    print('validation set pre training loss:', validation_loss)

    continue_training = True
    while continue_training:
        print('Epoch', epoch, '...')
        input_batches = []
        inputs, targets = train_dispenser.get_batch()
        train_loss = ctc_trainer.update(inputs, targets)
        print('loss:', train_loss)
        epoch = epoch + 1

        if epoch%5 == 0:
            print('\x1b[01;32m'
                  + "-----  validation Step --- time since start [h]   ",
                  (time.time() - start_time)/3600.0,
                  '\x1b[0m')
            epoch_loss_lst.append(train_loss)

            inputs, targets = val_dispenser.get_batch()
            validation_loss = ctc_trainer.evaluate(inputs, targets)
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
    ctc_trainer.save_model(filename)
    print("Model saved in file: %s" % filename)

    #run the network on the test data set.
    inputs, targets = test_dispenser.get_batch()
    test_loss = ctc_trainer.evaluate(inputs, targets)
    print("test loss: ", test_loss)

filename = "saved_models/" \
           + socket.gethostname() \
           + '-' + today \
           + ".pkl"
pickle.dump([epoch_loss_lst, epoch_loss_lst_val, test_loss, LEARNING_RATE,
             LEARNING_RATE_DECAY, epoch, general_settings, listener_settings],
             open(filename, "wb"))
print("plot values saved at: " + filename)

plt.plot(np.array(epoch_loss_lst))
plt.plot(np.array(epoch_loss_lst_val))
plt.show()

print("--- Program execution done --- time since start [h]",
      (time.time() - start_time)/3600.0)


