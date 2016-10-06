'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict phoneme sequences from n_features x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is basically a recreation of an experiment
on the TIMIT data set from chapter 7 of Alex Graves's book (Graves, Alex. Supervised Sequence
Labelling with Recurrent Neural Networks, volume 385 of Studies in Computational Intelligence.
Springer, 2012.), minus the early stopping.

Authors: mortiz wolter and Jon Rein
'''

#fix some pylint stuff
# fixes the np.random error
# pylint: disable=E1101
# fixes the unrecognized state_is_tuple
# pylint: disable=E1123

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#to store the data and name it properly.
import pickle
import socket

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from prepare.prepare_timit39 import load_batched_timit39


####Learning Parameters
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
OMEGA = 0.1 #weight regularization term.
INPUT_NOISE_STD = 0.6
#LEARNING_RATE = 0.0001       #too low?
#MOMENTUM = 0.6              #play with this.
MAX_N_EPOCHS = 1000
BATCH_COUNT = 8          #too small??
BATCH_COUNT_VAL = 1
BATCH_COUNT_TEST = 1

####Network Parameters
n_features = 40
n_hidden = 128
n_classes = 40 #39 phonemes, plus the "blank" for CTC

####Load data
print('Loading data')
data_batches = load_batched_timit39(BATCH_COUNT, BATCH_COUNT_VAL,
                                    BATCH_COUNT_TEST)
batched_data, max_time_steps, total, batch_size = data_batches[0]
batched_data_val, max_time_steps_val, total_val, batch_size_val = data_batches[1]
batched_data_test, max_time_steps_test, total_test, batch_size_test = data_batches[2]

#check if the padding has been done right.
assert max_time_steps == max_time_steps_val
assert max_time_steps == max_time_steps_test


#from IPython.core.debugger import Tracer
#Tracer()()

def create_dict(batched_data_arg, noise_bool):
    '''Create an input dictonary to be fed into the tree.
    @return:
    The dicitonary containing the input numpy arrays,
    the three sparse vector data components and the
    sequence legths of each utterance.'''

    batch_inputs, batch_trgt_sparse, batch_seq_lengths = batched_data_arg
    batch_trgt_ixs, batch_trgt_vals, batch_trgt_shape = batch_trgt_sparse
    res_feed_dict = {input_x: batch_inputs,
                     target_ixs: batch_trgt_ixs,
                     target_vals: batch_trgt_vals,
                     target_shape: batch_trgt_shape,
                     seq_lengths: batch_seq_lengths,
                     noise_wanted: noise_bool}
    return res_feed_dict, batch_seq_lengths

def blstm(input_list_fun, weights_blstm_fun, biases_blstm_fun):
    '''Define a simple bidicretional blstm layer with linear
       output nodes.'''

    initializer = tf.random_normal_initializer(0.0, 0.1)
    #initializer = tf.random_normal_initializer(0.0,np.sqrt(2.0 / (2*n_hidden)))
    initializer = None
    forward_h1 = rnn_cell.LSTMCell(n_hidden,
                                   use_peepholes=True,
                                   state_is_tuple=True,
                                   initializer=initializer)
    backward_h1 = rnn_cell.LSTMCell(n_hidden,
                                    use_peepholes=True,
                                    state_is_tuple=True,
                                    initializer=initializer)
    #compute the bidirectional RNN output throw away the states.
    #the output is a length T list consiting of
    # ([time][batch][cell_fw.output_size + cell_bw.output_size]) tensors.
    list_h1, _, _ = bidirectional_rnn(forward_h1, backward_h1, input_list_fun,
                                      dtype=tf.float32, scope='BDLSTM_H1')

    blstm_logits = [tf.matmul(T, weights_blstm_fun) + biases_blstm_fun for T in list_h1]

    print("length logit list:", len(blstm_logits))
    print("logit list element shape:", tf.Tensor.get_shape(blstm_logits[0]))
    #blstm_logits = [tf.nn.softmax(tf.matmul(T, weights_blstm_fun) +
    #                biases_blstm_fun) for T in list_h1]
    #blstm_logits = [tf.nn.softmax(T) for T in blstm_logits]
    return blstm_logits

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    #Variable wich determines if the graph is for training (it true add noise)
    noise_wanted = tf.placeholder(tf.bool, shape=[], name='add_noise')

    #### Graph input shape=(max_time_steps, batch_size, n_features),  but the first two change.
    input_x = tf.placeholder(tf.float32, shape=(max_time_steps, None, n_features))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #Split to get a list of 'n_steps' tensors of shape (batch_size, n_features)
    input_list = tf.unpack(input_x, num=max_time_steps, axis=0)
    #Target indices, values and shape used to create a sparse tensor.
    target_ixs = tf.placeholder(tf.int64, shape=None)    #indices
    target_vals = tf.placeholder(tf.int32, shape=None)   #vals
    target_shape = tf.placeholder(tf.int64, shape=None)  #shape
    target_y = tf.SparseTensor(target_ixs, target_vals, target_shape)
    seq_lengths = tf.placeholder(tf.int32, shape=None)

    #### Weights & biases
    weights_blstm = tf.Variable(tf.random_normal([n_hidden*2, n_classes],
                                                 mean=0.0, stddev=0.1,
                                                 dtype=tf.float32, seed=None,
                                                 name=None))
    #weights_blstm = tf.Variable(tf.truncated_normal([n_hidden*2, n_classes],
    #                                               stddev=np.sqrt(2.0 / (2*n_hidden))))
    biases_blstm = tf.Variable(tf.zeros([n_classes]))

    #determine if noise is wanted in this tree.
    def add_noise():
        '''Operation used add noise during training'''
        return [tf.random_normal(tf.shape(T), 0.0, INPUT_NOISE_STD)
                + T for T in input_list]
    def do_nothing():
        '''Operation used to select noise free inputs during validation
        and testing'''
        return input_list
    # tf cond applys the first operation if noise_wanted is true and
    # does nothing it the variable is false.
    blstm_input_list = tf.cond(noise_wanted, add_noise, do_nothing)

    #### Network
    logits = blstm(blstm_input_list, weights_blstm, biases_blstm)
    #### Optimizing
    # logits3d (max_time_steps, batch_size, n_classes), pack puts the list into a big matrix.
    #add the weight and bias l2 norms to the loss.
    trainable_weights = tf.trainable_variables()
    weight_loss = 0
    for trainable in trainable_weights:
        weight_loss += tf.nn.l2_loss(trainable)

    logits3d = tf.pack(logits)
    print("logits 3d shape:", tf.Tensor.get_shape(logits3d))
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, target_y, seq_lengths)) + OMEGA*weight_loss
    uncapped_optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)#.minimize(loss)
    #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    #gradient clipping:
    gvs = uncapped_optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    optimizer = uncapped_optimizer.apply_gradients(capped_gvs)

    #### Evaluating
    logits_max_test = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seq_lengths[0], 1])
    #predictions = ctc.ctc_beam_search_decoder(logits3d, seq_lengths, beam_width = 100)
    predictions = ctc.ctc_greedy_decoder(logits3d, seq_lengths)
    print("predictions", type(predictions))
    print("predictions[0]", type(predictions[0]))
    print("len(predictions[0])", len(predictions[0]))
    print("predictions[0][0]", type(predictions[0][0]))
    hypothesis = tf.to_int32(predictions[0][0])

    error_rate = tf.reduce_mean(tf.edit_distance(hypothesis, target_y, normalize=True))

#from IPython.core.debugger import Tracer
#Tracer()()

####Run session
restarts = 0
epoch_loss_lst = []
epoch_error_lst = []
epoch_error_lst_val = []

with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()

    #check untrained performance.
    batch_loss = np.zeros(len(batched_data))
    batch_errors = np.zeros(len(batched_data))
    batch_rand_ixs = np.array(range(0, len(batched_data)))
    for batch, batchOrigI in enumerate(batch_rand_ixs):
        feed_dict, batchSeqLengths = create_dict(batched_data[batchOrigI], True)
        l, wl, er, lmt = session.run([loss, weight_loss,
                                      error_rate, logits_max_test],
                                     feed_dict=feed_dict)
        print(np.unique(lmt)) #unique argmax values of first sample in batch;
        # should be blank for a while, then spit out target values
        if (batch % 1) == 0:
            print('Minibatch', batch, '/', batchOrigI, 'loss:', l, "weight loss:", wl)
            print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
        batch_errors[batch] = er*len(batchSeqLengths)
        batch_loss[batch] = l*len(batchSeqLengths)
    epoch_error_rate = batch_errors.sum() / total
    epoch_error_lst.append(epoch_error_rate)
    epoch_loss_lst.append(l.sum()/total)
    print('Untrained error rate:', epoch_error_rate)

    feed_dict, _ = create_dict(batched_data_val[0], False)
    vl, ver = session.run([loss, error_rate], feed_dict=feed_dict)
    print("untrained validation loss: ", vl, " validation error rate", ver)
    epoch_error_lst_val.append(ver)

    continue_training = True
    while continue_training:
        epoch = len(epoch_error_lst_val)
        print("params:", LEARNING_RATE, MOMENTUM, OMEGA, INPUT_NOISE_STD)
        print('Epoch', epoch, '...')
        batch_errors = np.zeros(len(batched_data))
        batch_rand_ixs = np.random.permutation(len(batched_data)) #randomize batch order
        for batch, batchOrigI in enumerate(batch_rand_ixs):
            feed_dict, batchSeqLengths = create_dict(batched_data[batchOrigI], True)
            _, l, wl, er, lmt = session.run([optimizer, loss, weight_loss,
                                             error_rate, logits_max_test],
                                            feed_dict=feed_dict)
            print(np.unique(lmt)) #print unique argmax values of first
                                  #sample in batch; should be
                                  #blank for a while, then spit
                                  #out target values
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l, "weight Loss", wl)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batch_errors[batch] = er*len(batchSeqLengths)
            batch_loss[batch] = l*len(batchSeqLengths)
        epoch_error_rate = batch_errors.sum() / total
        epoch_error_lst.append(epoch_error_rate)
        epoch_loss_lst.append(l.sum()/total)
        print('Epoch', epoch, 'error rate:', epoch_error_rate)
        #compute the validation error
        feed_dict, _ = create_dict(batched_data_val[0], False)
        vl, ver, vwl = session.run([loss, error_rate, weight_loss], feed_dict=feed_dict)
        print("vl: ", vl, " ver: ", "vwl: ", vwl)
        epoch_error_lst_val.append(ver)
        print("validation errors", epoch_error_lst_val)

        #stop if in the last 50 epochs no progress has been made.
        improvement_time = 100
        if epoch > improvement_time:
            min_last_50 = min(epoch_error_lst_val[(epoch - improvement_time):epoch])
            min_since_start = min(epoch_error_lst_val[0:(epoch - improvement_time)])
            if min_last_50 - 0.5 > (min_since_start):
                continue_training = False
                print("stopping the training.")

        if epoch > MAX_N_EPOCHS:
            continue_training = False

    #run the network on the test data set.
    feed_dict, _ = create_dict(batched_data_test[0], False)
    tl, ter = session.run([loss, error_rate], feed_dict=feed_dict)
    print("test loss: ", tl, " test error rate", ter)

filename = "saved/savedValsBLSTM." + socket.gethostname() + ".pkl"
pickle.dump([epoch_loss_lst, epoch_error_lst,
             epoch_error_lst_val, ter], open(filename, "wb"))
print("plot values saved at: " + filename)

plt.plot(np.array(epoch_loss_lst)/100.0)
plt.plot(np.array(epoch_error_lst))
plt.plot(np.array(epoch_error_lst_val))
plt.show()
