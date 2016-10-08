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
