
# fix the pylint import problem.
# pylint: disable=E0401

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from custompython.lazy_decorator import lazy_property

##Class for the training environment for a neural net graph
class Trainer(object):

    def __init__(self, model, learning_rate, omega):
        '''
        NnetTrainer constructor, creates the training graph
        #
        #@param model an nnetgraph object for the neural net
                 that will be used for decoding
        #@param learning_rate the initial learning rate
        '''
        #store the network graph
        self.learning_rate = learning_rate
        self.omega = omega
        self.model = model
        self.weight_loss = 0
        with self.model.tf_graph.as_default():
            #define additional placeholders related to training in the graph.
            #Target indices, values and shape used to create a sparse tensor.
            self.target_ixs = tf.placeholder(tf.int64, shape=None)    #indices
            self.target_vals = tf.placeholder(tf.int32, shape=None)   #vals
            self.target_shape = tf.placeholder(tf.int64, shape=None)  #shape
            self.target = tf.SparseTensor(self.target_ixs, self.target_vals,
                                          self.target_shape)

            # Make sure all properties are added to the trainer object upon
            # initialization.
            # pylint: disable=W0104
            self.loss
            self.optimizer = self.clipped_optimizer

    @lazy_property
    def loss(self):
        ''' Compute the loss. The loss value serves as an entry point for
            the reverse mode algorithmic differentiation algorithm, which
            does the training.
        '''
        #compute the model output.
        logits = self.model()
        loss = self.compute_loss(self.target, logits)

        #add the weight and bias l2 norms to the loss.
        trainable_weights = tf.trainable_variables()
        self.weight_loss = 0
        for trainable in trainable_weights:
            self.weight_loss += tf.nn.l2_loss(trainable)

        loss = loss + self.omega*self.weight_loss

        #loss = tf.reduce_mean(ctc.ctc_loss(self.model.logits, self.target,
        #                                   self.model.seq_lengths)) \
        #                                   + self.omega*self.weight_loss
        return loss


    @lazy_property
    def clipped_optimizer(self):
        '''operation can be called to appy clipped gradient optimization.'''
        #### Optimizing
        uncapped_optimizer = tf.train.AdamOptimizer(self.learning_rate)
                                                         #.minimize(loss)

        #gradient clipping:
        gvs = uncapped_optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) \
                     for grad, var in gvs]
        optimizer = uncapped_optimizer.apply_gradients(capped_gvs)
        return optimizer

    #@lazy_property
    #def logits_max_test(self):
    #    return tf.slice(tf.argmax(self.model.logits, 2), [0, 0],
    #                    [self.model.seq_lengths[0], 1])

    #pylint: disable=R0201
    #method could be a function, but it would not be smart to have a compute
    #loss function for all trainers. It is thus made a class method here.
    def compute_loss(self, targets, logits):
        '''Creates the operation to compute the cross-enthropy loss for every
         input frame (if you want to have a different loss function,
         overwrite this method)

        @param targets TODO fill out
        @param logits  TODO fill out

        @return a tensor containing the losses
        '''
        return tf.nn.softmax_cross_entropy_with_logits(logits,
                                                       targets,
                                                       name='loss')


def update(self, batched_data_list, session):
    '''
    Use the trainer to apply a training data batch, compute the
    gradient and update the model in the trainer.
    This command must be executed from within a session.
    '''
    batch_losses = np.zeros(len(batched_data_list))
    batch_errors = np.zeros(len(batched_data_list))
    for no, batch in enumerate(batched_data_list):
        feed_dict, batch_seq_lengths = self.create_dict(batch, True)
        _, l, wl, er, lmt = session.run([self.optimizer, self.loss,
                                         self.weight_loss,
                                         self.error_rate,
                                         self.logits_max_test],
                                        feed_dict=feed_dict)
        print(np.unique(lmt)) #print unique argmax values of first
                              #sample in batch; should be
                              #blank for a while, then spit
                              #out target values
        if (no % 1) == 0:
            print('Minibatch loss:', l, "weight loss:", wl)
            print('Minibatch error rate:', er)
        batch_errors[no] = er
        batch_losses[no] = l
    epoch_error_rate = batch_errors.sum() / len(batched_data_list)
    epoch_loss = batch_losses.sum() / len(batched_data_list)

    return epoch_loss, epoch_error_rate


def evaluate(self, batched_data_list, session):
    '''
    Evaluate model performance without applying gradients and no input
    noise.
    '''
    batch_losses = np.zeros(len(batched_data_list))
    batch_errors = np.zeros(len(batched_data_list))
    for no, batch in enumerate(batched_data_list):
        feed_dict, batch_seq_lengths = self.create_dict(batch, False)
        l, wl, er = session.run([self.loss,
                                 self.weight_loss,
                                 self.error_rate],
                                feed_dict=feed_dict)
        if (no % 1) == 0:
            print('Minibatch loss:', l, "weight loss:", wl)
            print('Minibatch error rate:', er)
        batch_errors[no] = er
        batch_losses[no] = l
    eval_error_rate = batch_errors.sum() / len(batched_data_list)
    eval_loss = batch_losses.sum() / len(batched_data_list)
    return eval_loss, eval_error_rate


def create_dict(self, batch, noise_bool):
    '''Create an input dictonary to be fed into the tree.
    @return:
    The dicitonary containing the input numpy arrays,
    the three sparse vector data components and the
    sequence legths of each utterance.'''

    batch_inputs, batch_trgt_sparse, batch_seq_lengths = batch
    batch_trgt_ixs, batch_trgt_vals, batch_trgt_shape = batch_trgt_sparse
    res_feed_dict = {self.model.input_x: batch_inputs,
                     self.target_ixs: batch_trgt_ixs,
                     self.target_vals: batch_trgt_vals,
                     self.target_shape: batch_trgt_shape,
                     self.model.seq_lengths: batch_seq_lengths,
                     self.model.noise_wanted: noise_bool}
    return res_feed_dict, batch_seq_lengths

def initialize(self):
    tf.initialize_all_variables().run()
