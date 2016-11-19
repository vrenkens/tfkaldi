'''@file trainer.py
neural network trainer environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import neuralnetworks.classifiers.seq_convertors as seq_convertors
from IPython.core.debugger import Tracer; debug_here = Tracer()

class Trainer(object):
    '''General class outlining the training environment of a classifier.'''
    __metaclass__ = ABCMeta

    def __init__(self, classifier, input_dim, max_input_length,
                 max_target_length, init_learning_rate, learning_rate_decay,
                 num_steps, numutterances_per_minibatch):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            classifier: the neural net classifier that will be trained
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximal length of the input sequences
            max_target_length: the maximal length of the target sequences
            init_learning_rate: the initial learning rate
            learning_rate_decay: the parameter for exponential learning rate
                decay
            num_steps: the total number of steps that will be taken
            numutterances_per_minibatch: determines how many utterances are
                processed at a time to limit memory usage
        '''

        self.numutterances_per_minibatch = numutterances_per_minibatch
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        #create the graph
        self.graph = tf.Graph()

        #define the placeholders in the graph
        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[numutterances_per_minibatch,
                                   max_input_length,
                                   input_dim],
                name='inputs')

            #reference labels
            self.targets = tf.placeholder(
                tf.int32, shape=[numutterances_per_minibatch,
                                 max_target_length,
                                 1],
                name='targets')

            #split the 3D targets tensor in a list of batch_size*input_dim
            #tensors

            #the length of all the input sequences
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[numutterances_per_minibatch],
                name='input_seq_length')

            #the length of all the output sequences
            self.target_seq_length = tf.placeholder(
                tf.int32, shape=[numutterances_per_minibatch],
                name='output_seq_length')

            #compute the training outputs of the nnetgraph
            trainlogits, logit_seq_length, self.modelsaver, self.control_ops =\
                classifier(self.inputs, self.input_seq_length,
                           is_training=True, reuse=False, scope='Classifier',
                           targets=self.targets,
                           target_seq_length=self.target_seq_length)

            #compute the validation output of the nnetgraph
            logits, _, _, _ = classifier(
                self.inputs, self.input_seq_length,
                is_training=False, reuse=True, scope='Classifier',
                targets=self.targets, target_seq_length=self.target_seq_length)

            #get a list of trainable variables in the decoder graph
            params = tf.trainable_variables()

            #add the variables and operations to the graph that are used for
            #training

            #total number of steps
            nsteps = tf.constant(num_steps, dtype=tf.int32, name='num_steps')

            #the total loss of the entire batch
            batch_loss = tf.get_variable(
                'batch_loss', [], dtype=tf.float32,
                initializer=tf.constant_initializer(0), trainable=False)

            with tf.variable_scope('train_variables'):

                #the amount of steps already taken
                self.global_step = tf.get_variable(
                    'global_step', [], dtype=tf.int32,
                    initializer=tf.constant_initializer(0), trainable=False)

                #a variable to scale the learning rate (used to reduce the
                #learning rate in case validation performance drops)
                learning_rate_fact = tf.get_variable(
                    'learning_rate_fact', [],
                    initializer=tf.constant_initializer(1.0), trainable=False)

                #compute the learning rate with exponential decay and scale with
                #the learning rate factor
                learning_rate = tf.train.exponential_decay(
                    init_learning_rate, self.global_step, nsteps,
                    learning_rate_decay) * learning_rate_fact

                #create the optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate)

            #for every parameter create a variable that holds its gradients
            with tf.variable_scope('gradients'):
                grads = [tf.get_variable(
                    param.op.name, param.get_shape().as_list(),
                    initializer=tf.constant_initializer(0),
                    trainable=False) for param in params]

            with tf.name_scope('train'):
                #the total number of frames that are used in the batch
                num_frames = tf.get_variable(
                    name='num_frames', shape=[], dtype=tf.int32,
                    initializer=tf.constant_initializer(0), trainable=False)

                #operation to update num_frames
                #pylint: disable=E1101
                update_num_frames = num_frames.assign_add(
                    tf.reduce_sum(self.target_seq_length))

                #compute the training loss
                loss = self.compute_loss(
                    self.targets, trainlogits, logit_seq_length,
                    self.target_seq_length)

                #operation to half the learning rate
                self.halve_learningrate_op = learning_rate_fact.assign(
                    learning_rate_fact/2).op

                #create an operation to initialise the gradients
                self.init_grads = tf.initialize_variables(grads)

                #the operation to initialise the batch loss
                #pylint: disable=E1101
                self.init_loss = batch_loss.initializer

                #the operation to initialize the num_frames
                #pylint: disable=E1101
                self.init_num_frames = num_frames.initializer

                #compute the gradients of the batch
                batchgrads = tf.gradients(loss, params)

                #create an operation to update the batch loss
                #pylint: disable=E1101
                self.update_loss = batch_loss.assign_add(loss)

                #create an operation to update the gradients, the batch_loss
                #and do all other update ops
                #pylint: disable=E1101
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.update_gradients_op = tf.group(
                    *([grads[p].assign_add(batchgrads[p])
                       for p in range(len(grads)) if batchgrads[p] is not None]
                      + [self.update_loss] + update_ops + [update_num_frames]),
                    name='update_gradients')

                #create an operation to apply the gradients
                #average the gradients
                meangrads = [tf.div(grad, tf.cast(num_frames, tf.float32),
                                    name=grad.op.name) for grad in grads]

                #clip the gradients
                meangrads = [tf.clip_by_value(grad, -1., 1.)
                             for grad in meangrads]

                #apply the gradients
                self.apply_gradients_op = optimizer.apply_gradients(
                    [(meangrads[p], params[p]) for p in range(len(meangrads))],
                    global_step=self.global_step, name='apply_gradients')

            with tf.name_scope('valid'):
                #compute the validation loss
                valid_loss = self.compute_loss(
                    self.targets, logits, logit_seq_length,
                    self.target_seq_length)

                #operation to update the validation loss
                #pylint: disable=E1101
                self.update_valid_loss = tf.group(
                    *([batch_loss.assign_add(valid_loss), update_num_frames]))

            #operation to compute the average loss in the batch
            self.average_loss = batch_loss/tf.cast(num_frames, tf.float32)

            # add an operation to initialise all the variables in the graph
            self.init_op = tf.initialize_all_variables()

            #saver for the training variables
            self.saver = tf.train.Saver(tf.get_collection(
                tf.GraphKeys.VARIABLES, scope='train_variables'))

            #create the summaries for visualisation
            self.summary = tf.merge_summary(
                [tf.histogram_summary(val.name, val)
                 for val in params+meangrads]
                + [tf.scalar_summary('loss', self.average_loss)])


        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

        #start without visualisation
        self.summarywriter = None

    @abstractmethod
    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the loss, this is specific to each
        trainer

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the output sequences as a
                vector

        Returns:
            a scalar value containing the total loss
        '''

        raise NotImplementedError("Abstract method")

    def initialize(self):
        '''Initialize all the variables in the graph'''

        self.init_op.run() #pylint: disable=E1101

    def start_visualization(self, logdir):
        '''
        open a summarywriter for visualisation and add the graph

        Args:
            logdir: directory where the summaries will be written
        '''

        self.summarywriter = tf.train.SummaryWriter(logdir=logdir,
                                                    graph=self.graph)


    def padd_batch(self, inputs, targets):
        '''
        Convert data input lists to time minor batches.

        Args:
            inputs: the inputs to the neural net, this should be a list
                containing an NxF matrix for each utterance in the batch where
                N is the number of frames in the utterance
            targets: the targets for neural nnet, this should be
                a list containing an N-dimensional vector for each utterance

        Returns:
            Time minor inputs, and targets.
            padded_inputs: [batch_size, max_time,
                            feature_number]
            padded_targets: [batch_size, max_target_length]

        '''

        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]
        output_seq_length = [t.shape[0] for t in targets]

        #fill the inputs to have a round number of minibatches
        added_inputs = (inputs
                        + (len(inputs)%self.numutterances_per_minibatch)
                        *[np.zeros([self.max_input_length,
                                    inputs[0].shape[1]])])

        added_targets = (targets
                         + (len(targets)%self.numutterances_per_minibatch)
                         *[np.zeros(self.max_target_length)])

        input_seq_length = \
            (input_seq_length
             + ((len(input_seq_length)%self.numutterances_per_minibatch))*[0])

        output_seq_length = \
            (output_seq_length
             + ((len(output_seq_length)%self.numutterances_per_minibatch))*[0])

        #pad all the inputs and targets to the max_length and put them in
        #one array
        padded_inputs = np.array([np.append(
            i, np.zeros([self.max_input_length-i.shape[0], i.shape[1]]), 0)
                                  for i in added_inputs])
        padded_targets = np.array([np.append(
            t, np.zeros(self.max_target_length-t.shape[0]), 0)
                                   for t in added_targets])

        #transpose the inputs and targets so they fit time major, placeholders
        #debug_here()
        #padded_inputs = padded_inputs.transpose([1, 0, 2])
        #padded_targets = padded_targets.transpose()

        return padded_inputs, padded_targets, input_seq_length, \
               output_seq_length, len(added_inputs)


    def split_batch(self, k, padded_inputs, padded_targets, input_seq_length,
                    output_seq_length):
        """
        Split batch data into smaller minibatches.

        Args:
            k:  Minibatch number and loop counter.
            padded_inputs: The input feature data numpy array, containing the
                           entire batch of size [batch_size, max_input_time,
                           feature_no].
            padded_targets: The target data numpy array for the full batch.
                            Size [batch_size, max_target_length]
            input_seq_length: The length of input sequence utterances a list
                              of length [batch_size]
            output_seq_length: The length of the target sequences a list of
                               length [batch_size]

        Returns:
            batch_inputs: Minibatch input [mini_batch_size, max_input_time,
                          feature_no]
            batch_targets: Minibatch targets size [mini_batch_size,
                           max_target_length]
            batch_input_seq_length: Input sequence length list of length
                                    [mini_batch_size]
            batch_output_seq_length: Target sequence legnth list of length
                                     [mini_batch_size]
        """

        batch_inputs = \
            padded_inputs[k*self.numutterances_per_minibatch\
                            :(k+1)*self.numutterances_per_minibatch,
                          :, :]

        batch_targets = padded_targets[
            k*self.numutterances_per_minibatch:
            (k+1)*self.numutterances_per_minibatch, :]

        batch_input_seq_length = input_seq_length[
            k*self.numutterances_per_minibatch:
            (k+1)*self.numutterances_per_minibatch]

        batch_output_seq_length = output_seq_length[
            k*self.numutterances_per_minibatch:
            (k+1)*self.numutterances_per_minibatch]

        return batch_inputs, batch_targets, batch_input_seq_length, \
               batch_output_seq_length


    def update(self, inputs, targets):
        '''
        update the neural model with a batch or training data

        Args:
            inputs: the inputs to the neural net, this should be a list
                containing an NxF matrix for each utterance in the batch where
                N is the number of frames in the utterance
            targets: the targets for neural nnet, this should be
                a list containing an N-dimensional vector for each utterance

        Returns:
            the loss at this step
        '''

        padded_inputs, padded_targets, input_seq_length, \
               output_seq_length, added_input_length \
                = self.padd_batch(inputs, targets)

        #feed in the batches one by one and accumulate the gradients and loss
        loopint = int(added_input_length/self.numutterances_per_minibatch)
        for k in range(loopint):
            batch_inputs, batch_targets, batch_input_seq_length, \
                batch_output_seq_length = \
            self.split_batch(k, padded_inputs, padded_targets,
                             input_seq_length, output_seq_length)

            #pylint: disable=E1101
            self.update_gradients_op.run(
                feed_dict={self.inputs:batch_inputs,
                           self.targets:batch_targets[:, :, np.newaxis],
                           self.input_seq_length:batch_input_seq_length,
                           self.target_seq_length:batch_output_seq_length})

        #apply the accumulated gradients to update the model parameters and
        #evaluate the loss
        if self.summarywriter is not None:
            [loss, summary, _] = tf.get_default_session().run(
                [self.average_loss, self.summary, self.apply_gradients_op])

            #pylint: disable=E1101
            self.summarywriter.add_summary(summary,
                                           global_step=self.global_step.eval())

        else:
            [loss, _] = tf.get_default_session().run(
                [self.average_loss, self.apply_gradients_op])


        #reinitialize the gradients and the loss
        #pylint: disable=E1101
        self.init_grads.run()
        self.init_loss.run()
        self.init_num_frames.run()

        return loss

    def evaluate(self, inputs, targets):
        '''
        Evaluate the performance of the neural net

        Args:
            inputs: the inputs to the neural net, this should be a list
                containing NxF matrices for each utterance in the batch where
                N is the number of frames in the utterance
            targets: the one-hot encoded targets for neural nnet, this should be
                a list containing an NxO matrix for each utterance where O is
                the output dimension of the neural net

        Returns:
            the loss of the batch
        '''

        if inputs is None or targets is None:
            return None

        padded_inputs, padded_targets, input_seq_length, \
               output_seq_length, added_input_length \
                = self.padd_batch(inputs, targets)

        #feed in the batches one by one and accumulate the gradients and loss
        loopint = int(added_input_length/self.numutterances_per_minibatch)
        for k in range(loopint):
            batch_inputs, batch_targets, batch_input_seq_length, \
                batch_output_seq_length = \
            self.split_batch(k, padded_inputs, padded_targets,
                             input_seq_length, output_seq_length)


            #pylint: disable=E1101
            self.update_valid_loss.run(
                feed_dict={self.inputs:batch_inputs,
                           self.targets:batch_targets[:, :, np.newaxis],
                           self.input_seq_length:batch_input_seq_length,
                           self.target_seq_length:batch_output_seq_length})

        #get the loss
        loss = self.average_loss.eval()

        #reinitialize the loss
        self.init_loss.run()
        self.init_num_frames.run()

        return loss

    def halve_learning_rate(self):
        '''halve the learning rate'''

        self.halve_learningrate_op.run()

    def save_model(self, filename):
        '''
        Save the model

        Args:
            filename: path to the model file
        '''
        self.modelsaver.save(tf.get_default_session(), filename)

    def restore_model(self, filename):
        '''
        Load the model

        Args:
            filename: path where the model will be saved
        '''
        self.modelsaver.restore(tf.get_default_session(), filename)

    def save_trainer(self, filename):
        '''
        Save the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''
        self.modelsaver.save(tf.get_default_session(), filename)
        self.saver.save(tf.get_default_session(), filename + '_trainvars')

    def restore_trainer(self, filename):
        '''
        Load the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''
        self.modelsaver.restore(tf.get_default_session(), filename)
        self.saver.restore(tf.get_default_session(), filename + '_trainvars')



class LasTrainer(Trainer):
    '''A trainer that minimises the cross-enthropy loss, using sequential
        logits and targets.'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_enthropy_loss'):
            #one hot encode the targets
            target_shape = tf.Tensor.get_shape(targets)
            target_matrix = tf.reshape(targets, [int(target_shape[0]),
                                                 int(target_shape[1])])
            targets_one_hot = tf.one_hot(target_matrix,
                                         int(logits.get_shape()[2]))
            #compute the cross-enthropy loss
            return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                logits, targets_one_hot))



class CrossEnthropyTrainer(Trainer):
    '''A trainer that minimises the cross-enthropy loss, the output sequences
    must be of the same length as the input sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_enthropy_loss'):

            #convert to non sequential data
            nonseq_targets = seq_convertors.seq2nonseq(targets,
                                                       target_seq_length)
            nonseq_logits = seq_convertors.seq2nonseq(logits, logit_seq_length)

            #make a vector out of the targets
            nonseq_targets = tf.reshape(nonseq_targets, [-1])

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets,
                                        int(nonseq_logits.get_shape()[1]))

            #compute the cross-enthropy loss
            return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                nonseq_logits, nonseq_targets))


class CTCTrainer(Trainer):
    '''A trainer that minimises the CTC loss, the output sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the CTC loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        idx, vals, shape = self.target_tensor_to_sparse(targets, target_seq_length)
        sparse_targets = tf.cast(tf.SparseTensor(idx, vals, shape), tf.int32)
        return tf.reduce_sum(tf.nn.ctc_loss(tf.pack(logits), sparse_targets,
                                            logit_seq_length, time_major=False))


    def target_tensor_to_sparse(self, target_tensor, target_seq_length):
        '''Make tensorflow SparseTensor from target tensor of shape 
           [numutterances_per_minibatch, max_target_length, 1] with each element
           in the list being a list or array with the values of the target sequence
           (e.g., the integer values of a character map for an ASR target string)
        '''
        target_tensor = tf.squeeze(target_tensor)
        #target_list = tf.unpack(target_tensor)
        zero = tf.constant(0, dtype=tf.int32)
        non_zero_mask = tf.not_equal(tf.cast(target_tensor, tf.int32), zero)  
        indices = tf.where(non_zero_mask)   
        vals = tf.boolean_mask(target_tensor, non_zero_mask)
        shape = [self.numutterances_per_minibatch, self.max_target_length]
        
        idx = tf.cast(tf.convert_to_tensor(indices), tf.int64)
        vls = tf.cast(tf.convert_to_tensor(vals), tf.int64)
        shp = tf.cast(tf.convert_to_tensor(shape), tf.int64)
        return idx, vls, shp

