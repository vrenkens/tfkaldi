##@package nnetgraph
# contains the functionality to create neural network graphs and train/test it

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
import nnetlayer

##This an abstrace class defining a neural net
#class NnetGraph(object, metaclass=ABCMeta):
class NnetGraph(object):
    __metaclass__ = ABCMeta

    ##NnetGraph constructor
    #
    #@param name name of the neural network
    #@param args arguments that will be used as properties of the neural net
    #@param kwargs named arguments that will be used as properties of the neural net
    def __init__(self, name, *args, **kwargs):

        self.name = name

        if len(args) + len(kwargs) != len(self.fieldnames):
            raise TypeError('%s() expects %d arguments (%d given)' %(type(self).__name__, len(self.fieldnames), len(args) + len(kwargs)))

        for a in range(len(args)):
            exec('self.%s = args[a]' % self.fieldnames[a])

        for a in kwargs:
            if a not in self.fieldnames:
                raise TypeError('%s is an invalid keyword argument for %s()' % (a, type(self).__name__))

            exec('self.%s = kwargs[a]' % (a))

    ##Extends the graph with the neural net graph, this method should define the attributes: inputs, outputs, logits and saver.
    @abstractmethod
    def extendGraph(self):
        pass

    ##A list of strings containing the fielnames of the __init__ function
    @abstractproperty
    def fieldnames(self):
        pass

    ##Inputs placeholder
    @property
    def inputs(self):
        return self._inputs

    ##Outputs of the graph
    @property
    def outputs(self):
        return self._outputs

    ##Logits used for training
    @property
    def trainlogits(self):
        return self._trainlogits

    ##Logits used for evaluating (can be the same as training)
    @property
    def testlogits(self):
        return self._testlogits

    ##Saver of the model parameters
    @property
    def saver(self):
        return self._saver

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @trainlogits.setter
    def trainlogits(self, trainlogits):
        self._trainlogits = trainlogits

    @testlogits.setter
    def testlogits(self, testlogits):
        self._testlogits = testlogits

    @saver.setter
    def saver(self, saver):
        self._saver = saver



##This class is a graph for feedforward fully connected neural nets. It is initialised as folows: DNN(name, input_dim, output_dim, num_hidden_layers, num_hidden_units, transfername, l2_norm, dropout)
#    name: name of the DNN
#     input_dim: the input dimension
#    output_dim: the output dimension
#    num_hidden_layers: number of hiden layers in the DNN
#    layer_wise_init: Boolean that is true if layerwhise initialisation should be done
#    num_hidden_units: number of hidden units in every layer
#    transfername: name of the transfer function that is used
#    l2_norm: boolean that determines of l2_normalisation is used after every layer
#    dropout: the chance that a hidden unit is propagated to the next layer
class DNN(NnetGraph):

    ##Extend the graph with the DNN
    def extendGraph(self):

        with tf.variable_scope(self.name):

            #define the input data
            self.inputs = tf.placeholder(tf.float32, shape = [None, self.input_dim], name = 'inputs')

            #placeholder to set the state prior
            self.prior = tf.placeholder(tf.float32, shape = [self.output_dim], name = 'priorGate')

            #variable that holds the state prior
            stateprior = tf.get_variable('prior', self.output_dim, initializer=tf.constant_initializer(0), trainable=False)

            #variable that holds the state prior
            initialisedlayers = tf.get_variable('initialisedlayers', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)

            #operation to increment the number of layers
            self.addLayerOp = initialisedlayers.assign_add(1).op

            #operation to set the state prior
            self.setPriorOp = stateprior.assign(self.prior).op

            #create the layers
            layers = [None]*(self.num_hidden_layers+1)

            #input layer
            layers[0] = nnetlayer.FFLayer(self.input_dim, self.num_hidden_units, 1/np.sqrt(self.input_dim), 'layer0', self.transfername, self.l2_norm, self.dropout)

            #hidden layers
            for k in range(1,len(layers)-1):
             layers[k] = nnetlayer.FFLayer(self.num_hidden_units, self.num_hidden_units, 1/np.sqrt(self.num_hidden_units), 'layer' + str(k), self.transfername, self.l2_norm, self.dropout)

            #output layer
            layers[-1] = nnetlayer.FFLayer(self.num_hidden_units, self.output_dim, 0, 'layer' + str(len(layers)-1))

            #operation to initialise the final layer
            self.initLastLayerOp = tf.initialize_variables([layers[-1].weights, layers[-1].biases])

            #do the forward computation with dropout

            activations = [None]*(len(layers)-1)
            activations[0]= layers[0](self.inputs)
            for l in range(1,len(activations)):
                activations[l] = layers[l](activations[l-1])

            if self.layer_wise_init:
                #compute the logits by selecting the activations at the layer that has last been added to the network, this is used for layer by layer initialisation
                self.trainlogits = layers[-1](tf.case([(tf.equal(initialisedlayers, tf.constant(l)), CallableTensor(activations[l])) for l in range(len(activations))], CallableTensor(activations[-1]),name = 'layerSelector'))
            else:
                self.trainlogits = layers[-1](activations[-1])

            if self.dropout<1:

                #do the forward computation without dropout

                activations = [None]*(len(layers)-1)
                activations[0]= layers[0](self.inputs, False)
                for l in range(1,len(activations)):
                    activations[l] = layers[l](activations[l-1], False)

                if self.layer_wise_init:
                    #compute the logits by selecting the activations at the layer that has last been added to the network, this is used for layer by layer initialisation
                    self.testlogits = layers[-1](tf.case([(tf.equal(initialisedlayers, tf.constant(l)), CallableTensor(activations[l])) for l in range(len(activations))], CallableTensor(activations[-1]),name = 'layerSelector'), False)
                else:
                    self.testlogits = layers[-1](activations[-1], False)
            else:
                self.testlogits = self.trainlogits

            #define the output
            self.outputs = tf.nn.softmax(self.testlogits)/stateprior

            #create a saver
            self.saver = tf.train.Saver()

    ##set the prior in the graph
    #
    #@param prior the state prior probabilities
    def setPrior(self, prior):
        self.setPriorOp.run(feed_dict={self.prior:prior})

    ##Add a layer to the network
    def addLayer(self):
        #reinitialise the final layer
        self.initLastLayerOp.run()

        #increment the number of layers
        self.addLayerOp.run()

    @property
    def fieldnames(self):
        return ['input_dim', 'output_dim', 'num_hidden_layers',  'layer_wise_init', 'num_hidden_units', 'transfername', 'l2_norm', 'dropout']

##Class for the decoding environment for a neural net graph
class NnetDecoder(object):
    ##NnetDecoder constructor, creates the decoding graph
    #
    #@param nnetgraph an nnetgraph object for the neural net that will be used for decoding
    def __init__(self, nnetGraph):

        self.graph = tf.Graph()
        self.nnetGraph = nnetGraph

        with self.graph.as_default():

            #create the decoding graph
            self.nnetGraph.extendGraph()

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

    ##decode using the neural net
    #
    #@param inputs the inputs to the graph as a NxF numpy array where N is the number of frames and F is the input feature dimension
    #
    #@return an NxO numpy array where N is the number of frames and O is the neural net output dimension
    def __call__(self, inputs):
        return self.nnetGraph.outputs.eval(feed_dict = {self.nnetGraph.inputs:inputs})

    ##load the saved neural net
    #
    #@param filename location where the neural net is saved
    def restore(self, filename):
        self.nnetGraph.saver.restore(tf.get_default_session(), filename)



##A class for a tensor that is callable
class CallableTensor:
    ##CallableTensor constructor
    #
    #@param tensor a tensor
    def __init__(self, tensor):
        self.tensor = tensor
    ##get the tensor
    #
    #@return the tensor
    def __call__(self):
        return self.tensor
