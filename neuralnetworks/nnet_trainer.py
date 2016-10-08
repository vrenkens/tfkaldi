
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

##Class for the training environment for a neural net graph
class Trainer(object):

    #NnetTrainer constructor, creates the training graph
    #
    #@param nnetgraph an nnetgraph object for the neural net that will be used for decoding
    #@param init_learning_rate the initial learning rate
    #@param learning_rate_decay the parameter for exponential learning rate decay
    #@param num_steps the total number of steps that will be taken
    def __init__(self, nnetGraph, LEARNING_RATE,
                 LEARNING_RATE_DECAY, INPUT_NOISE_STD, OMEGA):

        #store the network graph
        self.nnetGraph = nnetGraph

        #define the placeholders in the graph
        with self.nnetGraph.tf_graph.as_default():

            self.nnetGraph.extendGraph()

            #Target indices, values and shape used to create a sparse tensor.
            self.target_ixs = tf.placeholder(tf.int64, shape=None)    #indices
            self.target_vals = tf.placeholder(tf.int32, shape=None)   #vals
            self.target_shape = tf.placeholder(tf.int64, shape=None)  #shape
            target = tf.SparseTensor(self.target_ixs, self.target_vals,
                                     self.target_shape)

            #### Optimizing
            # logits3d (max_time_steps, batch_size, n_classes),
            # pack puts the list into a big matrix.
            #add the weight and bias l2 norms to the loss.
            trainable_weights = tf.trainable_variables()
            self.weight_loss = 0
            for trainable in trainable_weights:
                self.weight_loss += tf.nn.l2_loss(trainable)

            self.loss = tf.reduce_mean(ctc.ctc_loss(nnetGraph.logits3d, target,
                                               self.nnetGraph.seq_lengths)) \
                                               + OMEGA*self.weight_loss
            #uncapped_optimizer = tf.train.MomentumOptimizer(LEARNING_RATE,
                                                    # MOMENTUM)#.minimize(loss)
            uncapped_optimizer = tf.train.AdamOptimizer(LEARNING_RATE) #.minimize(loss)

            #gradient clipping:
            gvs = uncapped_optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.optimizer = uncapped_optimizer.apply_gradients(capped_gvs)

            #### Evaluating
            self.logits_max_test = tf.slice(tf.argmax(nnetGraph.logits3d, 2), [0, 0],
                                      [self.nnetGraph.seq_lengths[0], 1])
            #predictions = ctc.ctc_beam_search_decoder(logits3d, seq_lengths, beam_width = 100)
            self.error_rate = tf.reduce_mean(tf.edit_distance(self.nnetGraph.hypothesis,
                                                         target,
                                                         normalize=True))
