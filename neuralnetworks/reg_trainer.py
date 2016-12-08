'''@file trainer.py
neural network trainer environment'''
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from classifiers import seq_convertors
from IPython.core.debugger import Tracer; debug_here = Tracer();

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


            #the length of all the input sequences
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[numutterances_per_minibatch],
                name='input_seq_length')

            #the length of all the output sequences
            self.target_seq_length = tf.placeholder(
                tf.int32, shape=[numutterances_per_minibatch],
                name='output_seq_length')

            #compute the training outputs of the classifier
            train_logits, train_logit_seq_length, self.modelsaver, self.control_ops =\
                classifier(
                    self.inputs, self.input_seq_length, targets=self.targets,
                    target_seq_length=self.target_seq_length, is_training=True,
                    decoding=False, reuse=None, scope='Classifier')
  
            # use in the future to create a training validation graph
            # without dropout. 
            # compute the validation output of the classifier
            # val_logits, val_logit_seq_length, _, _ = classifier(
            #    self.inputs, self.input_seq_length, targets=self.targets,
            #    target_seq_length=self.target_seq_length, is_training=False,
            #    decoding=False, reuse=True, scope='Classifier')

            #compute the decoding output of the classifier
            dec_logits, dec_logit_seq_length, _, _ = classifier(
                self.inputs, self.input_seq_length, targets=self.targets,
                target_seq_length=self.target_seq_length, is_training=False,
                decoding=True, reuse=True, scope='Classifier')

            #get a list of trainable variables in the decoder graph
            params = tf.trainable_variables()

            #for every parameter create a variable that holds its gradients
            with tf.variable_scope('batch_variables'):

                with tf.variable_scope('gradients'):
                    #the gradients for the entire batch
                    grads = [tf.get_variable(
                        param.op.name, param.get_shape().as_list(),
                        initializer=tf.constant_initializer(0),
                        trainable=False) for param in params]

                #the total loss of the entire batch
                batch_loss = tf.get_variable(
                    'batch_loss', [], dtype=tf.float32,
                    initializer=tf.constant_initializer(0), trainable=False)

                #the total number of utterances that are used in the batch
                num_frames = tf.get_variable(
                    name='num_frames', shape=[], dtype=tf.int32,
                    initializer=tf.constant_initializer(0), trainable=False)

                #create an operation to initialise the batch variables
                self.init_batch = tf.initialize_variables(
                    grads + [num_frames, batch_loss])


            with tf.variable_scope('train'):

                #a variable to hold the amount of steps already taken
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
                self.learning_rate = tf.train.exponential_decay(
                    init_learning_rate, self.global_step, num_steps,
                    learning_rate_decay) * learning_rate_fact

                #create the optimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                #average the gradients
                meangrads = [tf.div(grad, tf.cast(num_frames, tf.float32),
                                    name=grad.op.name) for grad in grads]

                with tf.variable_scope('clip'):
                    #clip the gradients
                    meangrads = [tf.clip_by_value(grad, -1., 1.,
                                                  name=grad.op.name)
                                 for grad in meangrads]

                #opperation to apply the gradients
                self.apply_gradients_op = optimizer.apply_gradients(
                    [(meangrads[p], params[p]) for p in range(len(meangrads))],
                    global_step=self.global_step, name='apply_gradients')

                #compute the training loss
                loss = self.compute_loss(
                    self.targets, train_logits, train_logit_seq_length,
                    self.target_seq_length)


                #operation to half the learning rate
                self.halve_learningrate_op = learning_rate_fact.assign(
                    learning_rate_fact/2).op

                #compute the gradients of the batch
                batchgrads = tf.gradients(loss, params)

                #operation to update the batch loss
                #pylint: disable=E1101
                update_loss = batch_loss.assign_add(loss)

                #operation to update num_frames
                #pylint: disable=E1101
                update_num_frames = num_frames.assign_add(
                    numutterances_per_minibatch)

                #operation to update the gradients
                update_gradients = [
                    grads[p].assign_add(batchgrads[p])
                    for p in range(len(grads)) if batchgrads[p] is not None]

                #all remaining operations with the UPDATE_OPS GraphKeys
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                #create an operation to update the gradients, the batch_loss
                #and do all other update ops
                #pylint: disable=E1101
                self.update_op = tf.group(
                    *(update_gradients + [update_loss] + update_ops
                      + [update_num_frames]),
                    name='update_gradients')

                self.val_update_op = tf.group(
                    *([update_loss] + [update_num_frames]),
                    name='val_loss_update')

                #operation to compute the average loss in the batch
                self.average_loss = batch_loss/tf.cast(num_frames, tf.float32)

                #saver for the training variables
                self.saver = tf.train.Saver(tf.get_collection(
                    tf.GraphKeys.VARIABLES, scope='train'))

            with tf.name_scope('valid'):
                #compute the outputs that will be used for validation
                self.dec_outputs, self.dec_outputs_seq_length = \
                    self.validation(dec_logits, dec_logit_seq_length) 

            # add an operation to initialise all the variables in the graph
            self.init_op = tf.initialize_all_variables()

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
    def compute_loss(self, targets, logits, train_logit_seq_length,
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
            train_logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the output sequences as a
                vector

        Returns:
            a scalar value containing the total loss
        '''

        raise NotImplementedError('Abstract method')

    @abstractmethod
    def validation(self, val_logits, val_logit_seq_length):
        '''
        compute the outputs that will be used for validation

        Args:
            logits: the classifier output logits
            logit_seq_length: the sequence lengths of the logits

        Returns:
            The validation outputs, and output sequence lengths.
        '''

        raise NotImplementedError('Abstract method')

    @abstractmethod
    def validation_metric(self, outputs, output_seq_length, targets):
        '''
        compute the metric that will be used for validation

        The metric can be e.g. an error rate or a loss

        Args:
                      outputs: the validation outputs
                      targets: the ground truth labels
            output_seq_length: the output seq_lengths
        Returns:
            The validation outputs
        '''

        raise NotImplementedError('Abstract method')

    def initialize(self):
        '''Initialize all the variables in the graph'''

        self.init_op.run() #pylint: disable=E1101
        print("trainer initialized...")

    def start_visualization(self, logdir):
        '''
        open a summarywriter for visualisation and add the graph

        Args:
            logdir: directory where the summaries will be written
        '''

        self.summarywriter = tf.train.SummaryWriter(logdir=logdir,
                                                    graph=self.graph)


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
            a pair containing:
                - the loss at this step
                - the learning rate used at this step
        '''

        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]
        output_seq_length = [t.shape[0] for t in targets]

        #pad the inputs and targets till the maximum lengths
        padded_inputs, padded_targets = padd_batch(
            inputs, targets, self.max_input_length, self.max_target_length)

        #divide the batch into minibatches
        minibatches = split_batch(
            padded_inputs, padded_targets, input_seq_length, output_seq_length,
            self.numutterances_per_minibatch)

        #feed in the minibatches one by one and accumulate the gradients and
        #loss
        for minibatch in minibatches:
            #pylint: disable=E1101
            self.update_op.run(
                feed_dict={self.inputs:minibatch[0],
                           self.targets:minibatch[1],
                           self.input_seq_length:minibatch[2],
                           self.target_seq_length:minibatch[3]})

        #apply the accumulated gradients to update the model parameters and
        #evaluate the loss
        if self.summarywriter is not None:
            [loss, summary, lr, _] = tf.get_default_session().run(
                [self.average_loss, self.summary, self.learning_rate,
                 self.apply_gradients_op])

            #pylint: disable=E1101
            self.summarywriter.add_summary(summary,
                                           global_step=self.global_step.eval())

        else:
            [loss, _, lr] = tf.get_default_session().run(
                [self.average_loss, self.learning_rate,
                 self.apply_gradients_op])

        #reinitialize the batch variables
        #pylint: disable=E1101
        self.init_batch.run()

        return loss, lr

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

        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]
        output_seq_length = [t.shape[0] for t in targets]

        #pad the inputs and targets till the maximum lengths
        padded_inputs, padded_targets = padd_batch(
            inputs, targets, self.max_input_length, self.max_target_length)

        #divide the batch into minibatches
        minibatches = split_batch(
            padded_inputs, padded_targets, input_seq_length, output_seq_length,
            self.numutterances_per_minibatch)

        #feed in the minibatches one by one and get the validation outputs
        outputs = []
        seq_lengths = []
        for minibatch in minibatches:
            feed_dict = {self.inputs:minibatch[0],
                         self.targets:minibatch[1],
                         self.input_seq_length:minibatch[2],
                         self.target_seq_length:minibatch[3]}
            tf.get_default_session().run(
                [self.val_update_op], feed_dict=feed_dict)

            feed_dict = {self.inputs:minibatch[0], 
                         self.input_seq_length:minibatch[2]}
            output, seq_length = tf.get_default_session().run(
                [self.dec_outputs, self.dec_outputs_seq_length], feed_dict=feed_dict)
            outputs += list(output)
            seq_lengths += list(seq_length)

        error = self.validation_metric(outputs[:len(targets)], seq_lengths, targets)

        val_loss = tf.get_default_session().run([self.average_loss])[0]
        return error, val_loss

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

class LasCrossEnthropyTrainer(Trainer):
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

        with tf.variable_scope('weight_loss'):
            trainable_weights = tf.trainable_variables()
            weight_loss = 0
            for trainable in trainable_weights:
                weight_loss += tf.nn.l2_loss(trainable)

        with tf.name_scope('cross_enthropy_loss'):

            #training starts at t=1.
            targets_t_one = targets[:, 1:, :]
            target_seq_length_t_one = target_seq_length - 1

            #convert to non sequential data
            nonseq_targets = seq_convertors.seq2nonseq(targets_t_one,
                                                       target_seq_length_t_one)
            nonseq_logits = seq_convertors.seq2nonseq(logits, logit_seq_length)

            #make a vector out of the targets
            nonseq_targets = tf.reshape(nonseq_targets, [-1])

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets,
                                        int(nonseq_logits.get_shape()[1]))

            #compute the cross-enthropy loss
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                nonseq_logits, nonseq_targets))

        return loss

    def validation(self, logits, logit_seq_length):
        '''
        Do nothing.

        Args:
            logits: a [batch_size, max_input_length, dim] tensor containing the
                logits
            logit_seq_length: the length of all the input sequences as a vector

        Returns:
            a tensor with the same shape as logits with the label probabilities
        '''
        #remove outputs after eos token using the seq_lengths.

        return logits, logit_seq_length

    @staticmethod
    def greedy_search(clean_output):
        """ Extract the targets char probability
        Args:
            clean_output: A list of shape [batch_size][max_time, no_labels]
        Returns:
            utterance_char_batches: A list of shape [batch_size][max_time]"""
        utterance_char_batches = []
        for batch in range(0, len(clean_output)):
            utterance_chars_nos = []
            for time in range(0, clean_output[batch].shape[0]):
                utterance_chars_nos.append(np.argmax(clean_output[batch][time, :]))
            utterance_char_batches.append(np.array(utterance_chars_nos))
        return utterance_char_batches

    @staticmethod
    def edit_distance(seq1, seq2):
        """ Calculate edit distance between sequences x and y using
            matrix dynamic programming.  Return distance.
            source, Ben Langmead:
            http://www.cs.jhu.edu/~langmea/resources/lecture_notes/dp_and_edit_dist.pdf
        """
        dmat = np.zeros((len(seq1)+1, len(seq2)+1), dtype=int)
        dmat[0, 1:] = range(1, len(seq2)+1)
        dmat[1:, 0] = range(1, len(seq1)+1)
        for i in range(1, len(seq1)+1):
            for j in range(1, len(seq2)+1):
                delt = 1 if seq1[i-1] != seq2[j-1] else 0
                dmat[i, j] = min(dmat[i-1, j-1]+delt, dmat[i-1, j]+1, dmat[i, j-1]+1)
        return dmat[len(seq1), len(seq2)]

    def validation_metric(self, outputs, out_seq_length, targets):
        '''the cross-enthropy

        Args:
            outputs: the validation output, which is a list containing the
                label probabilities of size [batch_size][max_input_length, dim].
            targets: a list containing the ground truth target labels
        '''
        #concatenate the minibatches.
        batch_size = len(targets)
        clean_outputs = []
        for batch_count in range(batch_size):
            clean_outputs.append(outputs[batch_count][0:(out_seq_length[batch_count]), :])
        decoded_outputs = self.greedy_search(clean_outputs)
        num_frames = 0.0
        tot_lev = 0.0
        for utt_no in range(0, len(decoded_outputs)):
            num_frames += targets[utt_no].size
            tot_lev = tot_lev + self.edit_distance(decoded_outputs[utt_no], 
                                                   targets[utt_no])
        norm_lev = tot_lev/num_frames

        print('Example targets:    ', targets[0])
        print('Greedy  targets:    ', decoded_outputs[0])
        print('Decoded  length:', out_seq_length[0])
        
        return norm_lev

class CTCTrainer(Trainer):
    '''A trainer that minimises the CTC loss, the output sequences'''
    def __init__(self, classifier, input_dim, max_input_length,
                 max_target_length, init_learning_rate, learning_rate_decay,
                 num_steps, numutterances_per_minibatch, beam_width):
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
            beam_width: the width of the beam used for validation
        '''

        self.beam_width = beam_width
        super(CTCTrainer, self).__init__(
            classifier, input_dim, max_input_length, max_target_length,
            init_learning_rate, learning_rate_decay, num_steps, \
              numutterances_per_minibatch)

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the CTC loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a [batch_size, max_target_length, 1] tensor containing the
                targets
            logits: a [batch_size, max_input_length, dim] tensor containing the
                inputs
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('CTC_loss'):

            #get the batch size
            batch_size = int(targets.get_shape()[0])

            #convert the targets into a sparse tensor representation
            indices = tf.concat(0, [tf.concat(
                1, [tf.expand_dims(tf.tile([s], [target_seq_length[s]]), 1),
                    tf.expand_dims(tf.range(target_seq_length[s]), 1)])
                                    for s in range(batch_size)])

            values = tf.reshape(
                seq_convertors.seq2nonseq(targets, target_seq_length), [-1])

            shape = [batch_size, int(targets.get_shape()[1])]

            sparse_targets = tf.SparseTensor(tf.cast(indices, tf.int64), values,
                                             shape)

            loss = tf.reduce_sum(tf.nn.ctc_loss(logits, sparse_targets,
                                                logit_seq_length,
                                                time_major=False))

        return loss

    def validation(self, logits, logit_seq_length):
        '''
        decode the validation set with CTC beam search

        Args:
            logits: a [batch_size, max_input_length, dim] tensor containing the
                inputs
            logit_seq_length: the length of all the input sequences as a vector

        Returns:
            a matrix containing the decoded labels with size
            [batch_size, max_decoded_length]
        '''

        #Convert logits to time major
        tm_logits = tf.transpose(logits, [1, 0, 2])

        #do the CTC beam search
        sparse_output, _ = tf.nn.ctc_beam_search_decoder(
            tf.pack(tm_logits), logit_seq_length, self.beam_width)

        #convert the output to dense tensors with -1 as default values
        dense_output = tf.sparse_tensor_to_dense(sparse_output[0],
                                                 default_value=-1)

        return dense_output, logit_seq_length


    def validation_metric(self, outputs, output_seq_length, targets):
        '''the Label Error Rate for the decoded labels

        Args:
            outputs: the validation output, which is a list containing the
                decoded labels of size [batch_size][max_decoded_length]. the
                output sequences are padded with -1
            targets: a list containing the ground truth target labels
        '''
        #remove the padding from the outputs
        trimmed_outputs = [o[np.where(o != -1)] for o in outputs]
        errors = 0
        for k, target in enumerate(targets):
            error_matrix = np.zeros([target.size + 1,
                                     trimmed_outputs[k].size + 1])

            error_matrix[:, 0] = np.arange(target.size + 1)
            error_matrix[0, :] = np.arange(trimmed_outputs[k].size + 1)

            for x in range(1, target.size + 1):
                for y in range(1, trimmed_outputs[k].size + 1):
                    error_matrix[x, y] = min([
                        error_matrix[x-1, y] + 1, error_matrix[x, y-1] + 1,
                        error_matrix[x-1, y-1] + (target[x-1] !=
                                                  trimmed_outputs[k][y-1])])

            errors += error_matrix[-1, -1]

        ler = errors/sum([target.size for target in targets])

        return ler

def padd_batch(inputs, targets, input_length, target_length):
    '''
    Pad the inputs and targets so they have the maximum length

    Args:
        inputs: the inputs to the neural net, this should be a list
            containing an NxF matrix for each utterance in the batch where
            N is the number of frames in the utterance and F is the input
            feature dimension
        targets: the targets for neural nnet, this should be
            a list containing an T-dimensional vector for each utterance where
            T is the sequence length of the targets
        input_length: the length that will be used for padding the inputs
        target_length: the length that will be used for padding the targets

    Returns:
        a pair containing:
            - the padded inputs
            - the padded targets
    '''

    padded_inputs = [np.append(
        i, np.zeros([input_length-i.shape[0], i.shape[1]]), 0)
                     for i in inputs]
    padded_targets = [np.append(
        t, np.zeros(target_length-t.shape[0]), 0)
                      for t in targets]

    return padded_inputs, padded_targets


def split_batch(inputs, targets, input_seq_length, output_seq_length,
                numutterances_per_minibatch):
    '''
    Split batch data into smaller minibatches.

    Args:
        inputs: a list of input features
        targets: a list of targets
        input_seq_length: a list of inputs sequence lengths
        target_seq_length: a list of target sequence lengths
        numutterances_per_minibatch: number of utterances per minibatch

    Returns:
        a list of quadruples (one for each minibatch) containing numpy arrays
        where the first dimension is the minibatch size:
            - the inputs
            - the targets
            - the inputs sequence lengths
            - the outputs sequence lengths
    '''

    #fill the inputs to have a round number of minibatches
    added_inputs = (inputs
                    + (len(inputs)%numutterances_per_minibatch)
                    *[np.zeros([inputs[0].shape[0],
                                inputs[0].shape[1]])])

    added_targets = (targets
                     + (len(targets)%numutterances_per_minibatch)
                     *[np.zeros(targets[0].size)])

    added_input_seq_length = \
        (input_seq_length
         + ((len(input_seq_length)%numutterances_per_minibatch))*[0])

    added_output_seq_length = \
        (output_seq_length
         + ((len(output_seq_length)%numutterances_per_minibatch))*[0])

    #create the minibatches
    minibatches = []

    for k in range(len(added_inputs)/numutterances_per_minibatch):

        minibatch_inputs = np.array(
            added_inputs[k*numutterances_per_minibatch:
                         (k+1)*numutterances_per_minibatch])

        minibatch_targets = np.array(
            added_targets[k*numutterances_per_minibatch:
                          (k+1)*numutterances_per_minibatch])

        minibatch_input_seq_length = np.array(added_input_seq_length[
            k*numutterances_per_minibatch:
            (k+1)*numutterances_per_minibatch])

        minibatch_output_seq_length = np.array(added_output_seq_length[
            k*numutterances_per_minibatch:
            (k+1)*numutterances_per_minibatch])

        minibatches.append((minibatch_inputs,
                            minibatch_targets[:, :, np.newaxis],
                            minibatch_input_seq_length,
                            minibatch_output_seq_length))

    return minibatches