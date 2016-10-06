
import tensorflow as tf

##Class for the training environment for a neural net graph
class NnetTrainer(object):

    #NnetTrainer constructor, creates the training graph
    #
    #@param nnetgraph an nnetgraph object for the neural net that will be used for decoding
    #@param init_learning_rate the initial learning rate
    #@param learning_rate_decay the parameter for exponential learning rate decay
    #@param num_steps the total number of steps that will be taken
    #@param numframes_per_batch determines how many frames are processed at a time to limit memory usage
    def __init__(self, nnetGraph, init_learning_rate, learning_rate_decay, num_steps, numframes_per_batch):

        self.numframes_per_batch = numframes_per_batch
        self.nnetGraph = nnetGraph

        #create the graph
        self.graph = tf.Graph()

        #define the placeholders in the graph
        with self.graph.as_default():

            #create the decoding graph
            self.nnetGraph.extendGraph()

            #reference labels
            self.targets = tf.placeholder(tf.float32, shape = [None, self.nnetGraph.trainlogits.get_shape().as_list()[1]], name = 'targets')

            #input for the total number of frames that are used in the batch
            self.num_frames = tf.placeholder(tf.float32, shape = [], name = 'num_frames')

            #get a list of trainable variables in the decoder graph
            params = tf.trainable_variables()

            #add the variables and operations to the graph that are used for training

            #compute the training loss
            self.loss = tf.reduce_sum(self.computeLoss(self.targets, self.nnetGraph.trainlogits))

            #compute the validation loss
            self.validLoss = tf.reduce_sum(self.computeLoss(self.targets, self.nnetGraph.testlogits))

            #total number of steps
            Nsteps = tf.constant(num_steps, dtype = tf.int32, name = 'num_steps')

            #the total loss of the entire batch
            batch_loss = tf.get_variable('batch_loss', [], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)

            #operation to update the validation loss
            self.updateValidLoss = batch_loss.assign_add(self.validLoss).op

            with tf.variable_scope('train_variables'):

                #the amount of steps already taken
                self.global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

                #a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
                learning_rate_fact = tf.get_variable('learning_rate_fact', [], initializer=tf.constant_initializer(1.0), trainable=False)

                #compute the learning rate with exponential decay and scale with the learning rate factor
                learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_step, Nsteps, learning_rate_decay) * learning_rate_fact

                #create the optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate)

            #for every parameter create a variable that holds its gradients
            with tf.variable_scope('gradients'):
                grads = [tf.get_variable(param.op.name, param.get_shape().as_list(), initializer=tf.constant_initializer(0), trainable=False) for param in params]

            with tf.name_scope('train'):
                #operation to half the learning rate
                self.halveLearningRateOp = learning_rate_fact.assign(learning_rate_fact/2).op

                #create an operation to initialise the gradients
                self.initgrads = tf.initialize_variables(grads)

                #the operation to initialise the batch loss
                self.initloss = batch_loss.initializer

                #compute the gradients of the batch
                batchgrads = tf.gradients(self.loss, params)

                #create an operation to update the batch loss
                self.updateLoss = batch_loss.assign_add(self.loss).op

                #create an operation to update the gradients and the batch_loss
                self.updateGradientsOp = tf.group(*([grads[p].assign_add(batchgrads[p]) for p in range(len(grads)) if batchgrads[p] is not None] + [self.updateLoss]), name='update_gradients')

                #create an operation to apply the gradients
                self.applyGradientsOp = optimizer.apply_gradients([(grads[p]/self.num_frames, params[p]) for p in range(len(grads))], global_step=self.global_step, name='apply_gradients')

                # add an operation to initialise all the variables in the graph
                self.initop = tf.initialize_all_variables()

                #operation to compute the average loss in the batch
                self.average_loss = batch_loss/self.num_frames

            #saver for the training variables
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='train_variables'))

            #create the summaries for visualisation
            self.summary = tf.merge_summary([tf.histogram_summary(val.name, val) for val in params+grads] + [tf.scalar_summary('loss', batch_loss)])


        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

        #start without visualisation
        self.summarywriter = None

    ##Creates the operation to compute the cross-enthropy loss for every input frame (if you want to have a different loss function, overwrite this method)
    #
    #@param targets a NxO tensor containing the reference targets where N is the number of frames and O is the neural net output dimension
    #@param logits a NxO tensor containing the neural network output logits where N is the number of frames and O is the neural net output dimension
    #
    #@return an N-dimensional tensor containing the losses for all the input frames where N is the number of frames
    def computeLoss(self, targets, logits):
        return tf.nn.softmax_cross_entropy_with_logits(logits, targets, name='loss')

    ##Initialize all the variables in the graph
    def initialize(self):
        self.initop.run()

    ##open a summarywriter for visualisation and add the graph
    #
    #@param logdir directory where the summaries will be written
    def startVisualization(self, logdir):
        self.summarywriter = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)

    ##update the neural model with a batch or training data
    #
    #@param inputs the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
    #@param targets the one-hot encoded targets for neural nnet, this should be an NxO matrix where O is the output dimension of the neural net
    #
    #@return the loss at this step
    def update(self, inputs, targets):

        #if numframes_per_batch is not set just process the entire batch
        if self.numframes_per_batch==-1 or self.numframes_per_batch>inputs.shape[0]:
            numframes_per_batch = inputs.shape[0]
        else:
            numframes_per_batch = self.numframes_per_batch

        #feed in the batches one by one and accumulate the gradients and loss
        for k in range(int(inputs.shape[0]/numframes_per_batch) + int(inputs.shape[0]%numframes_per_batch > 0)):
            batchInputs = inputs[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
            batchTargets = targets[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
            self.updateGradientsOp.run(feed_dict = {self.nnetGraph.inputs:batchInputs, self.targets:batchTargets})

        #apply the accumulated gradients to update the model parameters
        self.applyGradientsOp.run(feed_dict = {self.num_frames:inputs.shape[0]})

        #get the loss at this step
        loss = self.average_loss.eval(feed_dict = {self.num_frames:inputs.shape[0]})

        #if visualization has started add the summary
        if self.summarywriter is not None:
            self.summarywriter.add_summary(self.summary.eval(), global_step=self.global_step.eval())

        #reinitialize the gradients and the loss
        self.initgrads.run()
        self.initloss.run()

        return loss


    ##Evaluate the performance of the neural net
    #
    #@param inputs the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
    #@param targets the one-hot encoded targets for neural nnet, this should be an NxO matrix where O is the output dimension of the neural net
    #
    #@return the loss of the batch
    def evaluate(self, inputs, targets):

        if inputs is None or targets is None:
            return None

        #if numframes_per_batch is not set just process the entire batch
        if self.numframes_per_batch==-1 or self.numframes_per_batch>inputs.shape[0]:
            numframes_per_batch = inputs.shape[0]
        else:
            numframes_per_batch = self.numframes_per_batch

        #feed in the batches one by one and accumulate the loss
        for k in range(int(inputs.shape[0]/self.numframes_per_batch) + int(inputs.shape[0]%self.numframes_per_batch > 0)):
            batchInputs = inputs[k*self.numframes_per_batch:min((k+1)*self.numframes_per_batch, inputs.shape[0]), :]
            batchTargets = targets[k*self.numframes_per_batch:min((k+1)*self.numframes_per_batch, inputs.shape[0]), :]
            self.updateValidLoss.run(feed_dict = {self.nnetGraph.inputs:batchInputs, self.targets:batchTargets})

        #get the loss
        loss = self.average_loss.eval(feed_dict = {self.num_frames:inputs.shape[0]})

        #reinitialize the loss
        self.initloss.run()

        return loss


    ##halve the learning rate
    def halve_learning_rate(self):
        self.halveLearningRateOp.run()

    ##Save the model
    #
    #@param filename path to the model file
    def saveModel(self, filename):
        self.nnetGraph.saver.save(tf.get_default_session(), filename)

    ##Load the model
    #
    #@param filename path where the model will be saved
    def restoreModel(self, filename):
        self.nnetGraph.saver.restore(tf.get_default_session(), filename)

    ##Save the training progress (including the model)
    #
    #@param filename path where the model will be saved
    def saveTrainer(self, filename):
        self.nnetGraph.saver.save(tf.get_default_session(), filename)
        self.saver.save(tf.get_default_session(), filename + '_trainvars')

    ##Load the training progress (including the model)
    #
    #@param filename path where the model will be saved
    def restoreTrainer(self, filename):
        self.nnetGraph.saver.restore(tf.get_default_session(), filename)
        self.saver.restore(tf.get_default_session(), filename + '_trainvars')
