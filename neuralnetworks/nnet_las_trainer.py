
import numpy as np
import tensorflow as tf
from custompython.lazy_decorator import lazy_property
from IPython.core.debugger import Tracer; debug_here = Tracer();


class LasTrainer(object):
    '''
    Defines the las training environment.
    '''
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
        self.model = model #an instance of NnetGraph.
        self.weight_loss = 0

        # create a tensorflow graph instance, which will serve as the training
        # graph.
        self.graph = tf.Graph()

        with self.graph.as_default():

            #Set up a placeholder for the input mel-bank-features.
            self.inputs = tf.placeholder(tf.float32, shape=(
                self.model.max_time_steps, self.model.batch_size,
                self.model.mel_feature_no))
            self.input_list = tf.unpack(self.inputs,
                                        num=self.model.max_time_steps,
                                        axis=0)

            self.mel_seq_lengths = tf.placeholder(
                tf.int32, shape=[self.model.batch_size])

            # Set up a place holder for the target sequences.
            # The placeholder expects a shape of
            # [max_target_seq_length, batch_size, no_labels].
            # This is a Time major structure.
            self.targets = tf.placeholder(self.model.dtype, shape=(
                None, self.model.batch_size, self.model.target_label_no))


            #package the input_list to meet the interface.
            graph_inputs = self.input_list, self.mel_seq_lengths, self.targets

            self.trainlogits, self.modelsaver, _ = \
                model(graph_inputs, is_training=True, reuse=False)


            # Make sure all properties are added to the trainer object upon
            # initialization.
            # pylint: disable=W0104
            self.loss
            self.optimizer = self.clipped_optimizer

            self.summary_writer = tf.train.SummaryWriter(logdir='log',
                                                         graph=self.graph)
            # Create the summary operation. If passed to the session the
            # recorded values are returned and can be written to disk with the
            # summary writer.
            self.merged_summaries = tf.merge_all_summaries()

    @lazy_property
    def loss(self):
        ''' Compute the loss. The loss value serves as an entry point for
            the reverse mode algorithmic differentiation algorithm, which
            does the training.
        '''
        #compute the model output.
        logits = self.trainlogits
        loss = LasTrainer.compute_loss(self.targets, logits)

        #add the weight and bias l2 norms to the loss.
        #trainable_weights = tf.trainable_variables()
        #self.weight_loss = 0
        #for trainable in trainable_weights:
        #    self.weight_loss += tf.nn.l2_loss(trainable)
        #loss = loss + self.omega*self.weight_loss

        #record the loss for analysis in tensorboard.
        tf.scalar_summary('loss', loss)
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

    @staticmethod
    def compute_loss(targets, logits):
        '''Creates the operation to compute the cross-enthropy loss for every
         input frame (if you want to have a different loss function,
         overwrite this method)

        @param targets TODO fill out
        @param logits  TODO fill out

        @return a tensor containing the losses
        '''
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
            logits, targets, name='loss'))

    def update(self, batched_data_list, session):
        '''
        Use the trainer to apply a training data batch, compute the
        gradient and update the model in the trainer.
        This command must be executed from within a session.
        '''
        batch_losses = np.zeros(len(batched_data_list))
        batch_errors = np.zeros(len(batched_data_list))
        for no, batch in enumerate(batched_data_list):
            feed_dict = self.create_dict(batch)
            _, l = session.run([self.optimizer, self.loss],
                               feed_dict=feed_dict)
            if (no % 1) == 0:
                print('Batch loss:', l)
            batch_losses[no] = l
        epoch_error_rate = batch_errors.sum() / len(batched_data_list)
        epoch_loss = batch_losses.sum() / len(batched_data_list)
        return epoch_loss, epoch_error_rate


    def evaluate(self, batched_data_list, session, epoch):
        '''
        Evaluate model performance without applying gradients and no input
        noise.
        '''
        batch_losses = np.zeros(len(batched_data_list))
        for no, batch in enumerate(batched_data_list):
            feed_dict = self.create_dict(batch)
            l, merged = session.run([self.loss, self.merged_summaries],
                                    feed_dict=feed_dict)
            if (no % 1) == 0:
                print('Batch loss:', l)
                self.summary_writer.add_summary(merged, global_step=epoch)
            batch_losses[no] = l
        eval_loss = batch_losses.sum() / len(batched_data_list)
        return eval_loss

    def create_dict(self, batch):
        '''Create an input dictonary to be fed into the tree.
        @return:
        The dicitonary containing the input numpy arrays,
        the three sparse vector data components and the
        sequence legths of each utterance.'''
        batch_mel_inputs, batch_targets, batch_mel_lengths = batch
        res_feed_dict = {self.inputs: batch_mel_inputs,
                         self.targets: batch_targets,
                         self.mel_seq_lengths: batch_mel_lengths}
        return res_feed_dict

    def initialize(self):
        tf.initialize_all_variables().run()
