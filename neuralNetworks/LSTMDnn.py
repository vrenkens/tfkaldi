
#Set up an LSTM_DNN structure.
class LSTM_DNN(NnetGraph):

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
            #TODO: create lstm hidden layer parameter.
            layers[0] = nnetlayer.pLSTM(self.input_dim, self.num_hidden_units, 128)

             #hidden layers
            for k in range(1,len(layers)-1):
                 layers[k] = nnetlayer.pLSTM(self.num_hidden_units, self.num_hidden_units, 128)

             #output layer
            layers[-1] = nnetlayer.pLSTM(self.num_hidden_units, self.output_dim, 128)

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
