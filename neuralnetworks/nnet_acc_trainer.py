
# fix the pylint import problem.
# pylint: disable=E0401
# pylint: disable=F0401
# pylint: disable=W0104
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from custompython.lazy_decorator import lazy_property
from IPython.core.debugger import Tracer; debug_here = Tracer()

##Class for the training environment for a neural net graph
class AccumulationTrainer(object):
    '''Train a given model by accumulation gradients trough several mini-
      batches and applying them at the end.
    '''
    def __init__(self, model, learning_rate, omega, max_batch_size):
        #store the network graph
        self.learning_rate = learning_rate
        self.omega = omega
        self.model = model
        self.mxbtchsze = max_batch_size
        self.weight_loss = 0
        with self.model.tf_graph.as_default():
            #define additional placeholders related to training in the graph.
            #Target indices, values and shape used to create a sparse tensor.
            self.target_ixs = tf.placeholder(tf.int64, shape=None)    #indices
            self.target_vals = tf.placeholder(tf.int32, shape=None)   #vals
            self.target_shape = tf.placeholder(tf.int64, shape=None)  #shape
            self.target = tf.SparseTensor(self.target_ixs, self.target_vals,
                                          self.target_shape)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.trnbl_wghts = tf.trainable_variables()

            cnst_zero = tf.constant_initializer(0)
            with tf.variable_scope('train_variables'):
                self.global_step = tf.get_variable('global_step', [],
                                                   dtype=tf.int32,
                                                   initializer=cnst_zero,
                                                   trainable=False)


            #get a list of trainable variables in the training graph
            #for every parameter create a variable that holds its gradients
            with tf.variable_scope('gradients'):
                self.seq_length_sums = []
                self.batch_counter = tf.placeholder(dtype=tf.int32, shape=[],
                                                    name='batch_counter')
                self.grads = [tf.get_variable(weight.op.name,
                         weight.get_shape().as_list(),
                         initializer=tf.constant_initializer(0),
                         trainable=False) for weight in self.trnbl_wghts]

            # Make sure all properties are added to the trainer object upon
            # initialization.
            self.loss
            self.apply_gradients
            self.logits_max_test
            self.error_rate
            self.accumulate_gradients

    @lazy_property
    def accumulate_gradients(self):
        '''Run the batch data trough the network and upbate the
           the gradient candidate values, but dont apply them yet. '''
        length_sum = tf.cast(tf.reduce_sum(self.model.seq_lengths), tf.float32)
        self.seq_length_sums.append(length_sum)
        #TODO: what is the difference between optimizer.compute_gradients
        #      and tf.gradients?
        #batchgrads = self.optimizer.compute_gradients(self.loss,
        #                                              self.trnbl_wghts)
        batchgrads = tf.gradients(self.loss, self.trnbl_wghts)
        return tf.group(*([self.grads[p].assign_add(batchgrads[p]*length_sum)
                             for p in range(len(self.grads))
                             if batchgrads[p] is not None]
                             + [self.loss]),
                             name='update_gradients')

    @lazy_property
    def apply_gradients(self):
        '''
        capp and apply previously computed gradient values.
        '''
        #### Optimizing
        total_length_sum = tf.reduce_sum(tf.pack(self.seq_length_sums))
        #gradient scaling
        scl_grad = [grad / tf.cast(total_length_sum, tf.float32)
                    for grad in self.grads]
        #gradient clipping
        clip_grads = [tf.clip_by_value(grad, -1., 1.) for grad in scl_grad]
        optimizer = self.optimizer.apply_gradients( \
          [(clip_grads[p], self.trnbl_wghts[p]) for p in range(
            len(clip_grads))], global_step=self.global_step,
             name='apply_gradients')
        return optimizer

    @lazy_property
    def loss(self):
        ''' evaluate the ctc loss function.
            The edit distance is smallest number of changes required to go
            from one code to another.'''
        #add the weight and bias l2 norms to the loss.
        self.weight_loss = 0
        for trainable in self.trnbl_wghts:
            self.weight_loss += tf.nn.l2_loss(trainable)
        loss = tf.reduce_mean(ctc.ctc_loss(self.model.logits, self.target,
                                           self.model.seq_lengths)) \
                                           + self.omega*self.weight_loss
        return loss

    @lazy_property
    def logits_max_test(self):
        '''perform the logits max test to see which predictions have settled
           should be empty initially and eventually fill up with values.'''
        return tf.slice(tf.argmax(self.model.logits, 2), [0, 0],
                        [self.model.seq_lengths[0], 1])

    @lazy_property
    def error_rate(self):
        '''Compute the distance of the hypothesis to the target values.
           The edit distance is the smallest amound of changes required
           to get from one to the other.'''
        return tf.reduce_mean(tf.edit_distance(self.model.hypothesis,
                                               self.target,
                                               normalize=True))


    def update(self, batched_data_list, session):
        '''
        Use the trainer to apply a training data batch, compute the
        gradient and update the model in the trainer.
        This command must be executed from within a session.
        '''
        batch_losses = np.zeros(len(batched_data_list))
        batch_errors = np.zeros(len(batched_data_list))
        for batch_idx, batch in enumerate(batched_data_list):
            feed_dict, _ = self.create_dict(batch, True)
            _, loss, weight_loss, error_rate, lmt = \
                                 session.run([self.accumulate_gradients,
                                             self.loss,
                                             self.weight_loss,
                                             self.error_rate,
                                             self.logits_max_test],
                                            feed_dict=feed_dict)
            print(np.unique(lmt)) #print unique argmax values of first
                                  #sample in batch; should be
                                  #blank for a while, then spit
                                  #out target values
            if (batch_idx % 1) == 0:
                print('Minibatch loss:', loss, "weight loss:", weight_loss)
                print('Minibatch error rate:', error_rate)
            batch_errors[batch_idx] = error_rate
            batch_losses[batch_idx] = loss

        epoch_error_rate = batch_errors.sum() / len(batched_data_list)
        epoch_loss = batch_losses.sum() / len(batched_data_list)

        #apply the accumulated gradients.
        _ = session.run([self.apply_gradients])

        return epoch_loss, epoch_error_rate

    def evaluate(self, batched_data_list, session):
        '''
        Evaluate model performance without applying gradients and no input
        noise.
        '''
        batch_losses = np.zeros(len(batched_data_list))
        batch_errors = np.zeros(len(batched_data_list))
        for batch_idx, batch in enumerate(batched_data_list):
            feed_dict, _ = self.create_dict(batch, False)
            loss, weight_loss, error_rate = session.run([self.loss,
                                     self.weight_loss,
                                     self.error_rate],
                                    feed_dict=feed_dict)
            if (batch_idx % 1) == 0:
                print('Minibatch loss:', loss, "weight loss:", weight_loss)
                print('Minibatch error rate:', error_rate)
            batch_errors[batch_idx] = error_rate
            batch_losses[batch_idx] = loss
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
