##@package nnetlayer
# contains neural network layers

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

##This class defines a pyramidal lstm layer.
class pLSTM(object):

	##pyramidal LSTM constructor, defines the variables
	# output_dim = number of classes.
	def __init__(self, input_dim, output_dim, lstm_dim):
		with tf.variable_scope('forward'):
			self.forward_lstm_block = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
		with tf.variable_scope('backward'):
			self.backward_lstm_block = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)

		#create the model parameters in this layer
		with tf.variable_scope(name + '_parameters'):
			self.weights = tf.get_variable('weights', [2*lstm_dim, output_dim], initializer=tf.random_normal_initializer(stddev=weights_std))
			self.biases = tf.get_variable('biases',  [output_dim], initializer=tf.constant_initializer(0))

	def __call__(self, inputs, apply_dropout = True):
		outputs, output_state_fw, output_state_bw = bidirectional_rnn(self.forward_lstm_block,
			self.backward_lstm_block, inputs)

		#linear neuron computes the output.
		return tf.matmul(outputs[-1],self.weights) + self.biases

##This class defines a fully connected feed forward layer
class FFLayer(object):

	##FFLayer constructor, defines the variables.
	#
	#@param input_dim input dimension of the layer.
	#@param output_dim output dimension of the layer.
	#@param weights_std standart deviation of the weights initializer.
	#@param name name of the layer.
	#@param transfername name of the transfer function that is used.
	#@param l2_norm boolean that determines of l2_normalisation is used after every layer.
	#@param dropout the chance that a hidden unit is propagated to the next layer.
	def __init__(self, input_dim, output_dim, weights_std, name, transfername='linear', l2_norm=False, dropout=1):

		#create the model parameters in this layer
		with tf.variable_scope(name + '_parameters'):
			self.weights = tf.get_variable('weights', [input_dim, output_dim], initializer=tf.random_normal_initializer(stddev=weights_std))
			self.biases = tf.get_variable('biases',  [output_dim], initializer=tf.constant_initializer(0))

		#save the parameters
		self.transfername = transfername
		self.l2_norm = l2_norm
		self.dropout = dropout
		self.name = name

	## Do the forward computation.
	#
	#@param inputs the input to the layer
	#@param apply_dropout bool to determine if dropout is aplied
	#
	#@return the output of the layer
	def __call__(self, inputs, apply_dropout = True):

		with tf.name_scope(self.name):

			#apply weights and biases
			outputs = transferFunction(tf.matmul(inputs, self.weights) + self.biases, self.transfername)

			#apply l2 normalisation
			if self.l2_norm:
				outputs = transferFunction(outputs, 'l2_norm')

			#apply dropout
			if self.dropout<1 and apply_dropout:
				outputs = tf.nn.dropout(outputs, self.dropout)

		return outputs


##Apply the transfer function
#
#@param inputs the inputs to the transfer function
#@param name the name of the function, current options are: relu, sigmoid, tanh, linear or l2_norm
#
#@return the output to the transfer function
def transferFunction(inputs, name):
	if name == 'relu':
		return tf.nn.relu(inputs)
	elif name== 'sigmoid':
		return tf.nn.sigmoid(inputs)
	elif name == 'tanh':
		return tf.nn.tanh(inputs)
	elif name == 'linear':
		return inputs
	elif name == 'l2_norm':
		with tf.name_scope('l2_norm'):
			#compute the mean squared value
			sig = tf.reduce_mean(tf.square(inputs), 1, keep_dims=True)

			#divide the input by the mean squared value
			normalized = inputs/sig

			#if the mean squared value is larger then one select the normalized value otherwise select the unnormalised one
			return tf.select(tf.greater(tf.reshape(sig, [-1]), 1), normalized, inputs)
	else:
		raise Exception('unknown transfer function: %s' % name)
