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

import time
import datetime
import pickle
import socket
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from prepare.batch_dispenser import PhonemeTextDispenser
from prepare.batch_dispenser import UttTextDispenser
from prepare.feature_reader import FeatureReader
from neuralnetworks.nnet_las_model import LasModel
from neuralnetworks.nnet_las_trainer import LasTrainer
from IPython.core.debugger import Tracer; debug_here = Tracer()

start_time = time.time()

def generate_dispenser(data_path, set_kind, label_no, batch_size, phonemes):
    """ Instatiate a batch dispenser object using the data
        at the spcified path locations"""
    feature_path = data_path + set_kind + "/" + "feats.scp"
    cmvn_path = data_path + set_kind + "/" + "cmvn.scp"
    utt2spk_path = data_path + set_kind + "/" + "utt2spk"
    text_path = data_path + set_kind + "/" + "text"
    feature_reader = FeatureReader(feature_path, cmvn_path, utt2spk_path)

    if phonemes is True:
        dispenser = PhonemeTextDispenser(feature_reader, batch_size,
                                         text_path, label_no,
                                         max_time_steps, one_hot_encoding=True)
    else:
        dispenser = UttTextDispenser(feature_reader, batch_size,
                                     text_path, label_no,
                                     max_time_steps, one_hot_encoding=True)
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
OVERFIT_TOL = 99999

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

#BATCH_COUNT = train_dispenser.get_batch_count()
BATCH_COUNT = 5
BATCH_COUNT_VAL = val_dispenser.get_batch_count()
BATCH_COUNT_TEST = test_dispenser.get_batch_count()
print(BATCH_COUNT, BATCH_COUNT_VAL, BATCH_COUNT_TEST)
n_classes = AURORA_LABELS

test_batch = test_dispenser.get_batch()
#create the las arcitecture
las_model = LasModel(max_time_steps, MEL_FEATURE_NO, MAX_BATCH_SIZE,
                     AURORA_LABELS)
las_trainer = LasTrainer(las_model, LEARNING_RATE, OMEGA)

print("--- Tree generation done. --- time since start [s]",
     (time.time() - start_time))

####Run session
restarts = 0
epoch_loss_lst = []
epoch_loss_lst_val = []

print("Graph done, starting computation.")
with tf.Session(graph=las_trainer.graph) as session:
    print('Initializing')
    las_trainer.initialize()

    #check untrained performance.
    input_batches = []
    for batch in range(0, 2):
        input_batches.append(train_dispenser.get_batch())

    eval_loss = las_trainer.evaluate(input_batches, session, 1)

    #val_lst = [val_dispenser.get_batch()]
    #vl = las_trainer.evaluate(val_lst, session, 1)
    #print("untrained validation loss: ", vl)
    #epoch_loss_lst_val.append(vl)

    continue_training = True
    while continue_training:
        epoch = len(epoch_loss_lst_val)
        print('Epoch', epoch, '...')
        input_batches = []
        for batch in range(0, BATCH_COUNT):
            input_batches.append(train_dispenser.get_batch())
        trn_loss = las_trainer.update(input_batches, session)

        epoch_loss_lst.append(trn_loss)
        print('loss:', trn_loss)

        val_lst = [val_dispenser.get_batch()]
        vl = las_trainer.evaluate(val_lst, session, epoch+1)
        print("validation loss: ", vl)
        epoch_loss_lst_val.append(vl)

        # if the training error is lower than the validation error for
        # interval iterations stop..
        interval = 50
        if epoch > interval:
            print("validation errors",
                  epoch_loss_lst_val[(epoch - interval):epoch])

            val_mean = np.mean(epoch_loss_lst_val[(epoch - interval):epoch])
            train_mean = np.mean(epoch_loss_lst[(epoch - interval):epoch])
            test_val = val_mean - train_mean - OVERFIT_TOL
            print('Overfit condition value:', test_val,
                  'remaining iterations: ', MAX_N_EPOCHS - epoch)
            if (test_val > 0) or (epoch > MAX_N_EPOCHS):
                continue_training = False
                print("stopping the training.")
        else:
            print("validation losses", epoch_loss_lst_val, epoch)

    #run the network on the test data set.
    test_lst = [test_dispenser.get_batch()]
    tl = las_trainer.evaluate(test_lst, session, epoch+1)
    print("test loss: ", tl)

now = datetime.datetime.now()
filename = "saved/savedValsLastAdam." \
           + socket.gethostname() \
           + str(now) \
           + ".pkl"
pickle.dump([epoch_loss_lst, epoch_loss_lst_val, tl, LEARNING_RATE,
             MOMENTUM, OMEGA], open(filename, "wb"))
print("plot values saved at: " + filename)

plt.plot(np.array(epoch_loss_lst))
plt.plot(np.array(epoch_loss_lst_val))
plt.show()

print("--- Program execution done --- time since start [s]",
      (time.time() - start_time))
