
# fix the pylint import problem.
# pylint: disable=E0401

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from custompython.lazy_decorator import lazy_property

##Class for the training environment for a neural net graph
class Trainer(object):

    #NnetTrainer constructor, creates the training graph
    #
    #@param nnetgraph an nnetgraph object for the neural net that will be used for decoding
    #@param init_learning_rate the initial learning rate
    #@param learning_rate_decay the parameter for exponential learning rate decay
    #@param num_steps the total number of steps that will be taken
    def __init__(self, model, learning_rate,
                 learning_rate_DECAY, OMEGA):
        #store the network graph
        self.learning_rate = learning_rate
        self.OMEGA = OMEGA
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
            self.logits_max_test
            self.error_rate

    @lazy_property
    def loss(self):
        #add the weight and bias l2 norms to the loss.
        trainable_weights = tf.trainable_variables()
        self.weight_loss = 0
        for trainable in trainable_weights:
            self.weight_loss += tf.nn.l2_loss(trainable)

        loss = tf.reduce_mean(ctc.ctc_loss(self.model.logits, self.target,
                                           self.model.seq_lengths)) \
                                           + self.OMEGA*self.weight_loss
        return loss


    @lazy_property
    def clipped_optimizer(self):
        #### Optimizing
        #uncapped_optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                # MOMENTUM)#.minimize(loss)
        uncapped_optimizer = tf.train.AdamOptimizer(self.learning_rate) #.minimize(loss)

        #gradient clipping:
        gvs = uncapped_optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        optimizer = uncapped_optimizer.apply_gradients(capped_gvs)
        return optimizer

    @lazy_property
    def logits_max_test(self):
        return tf.slice(tf.argmax(self.model.logits, 2), [0, 0],
                               [self.model.seq_lengths[0], 1])

    @lazy_property
    def error_rate(self):
        return tf.reduce_mean(tf.edit_distance(self.model.hypothesis,
                                               self.target,
                                               normalize=True))
    
    @lazy_property
    def halve_learning_rate(self):
        self.learning_rate = self.learning_rate/2
