##@package feat 
#Contains the class for feature computation

import sys
sys.path.append('features/python_speech_features')
import base
import scipy.io.wavfile as wav
import numpy as np

## A featurecomputer is used to compute a certain type of features
class FeatureComputer(object):

	##FeatureComputer constructor
	#
	#@param featureType string containing the type of features, optione are 'fbank', 'mfcc' and 'ssc'
	#@param dynamic the type of dynamic information added, options are nodelta, delta and ddelta
	#@param conf the feature configuration
	def __init__(self, featureType, dynamic, conf):
		if featureType == 'fbank':
			self.compFeat = base.logfbank
		elif featureType == 'mfcc':
			self.compFeat = base.mfcc
		elif featureType == 'ssc':
			self.compFeat = base.ssc
		else:
			raise Exception('unknown feature type')
			
		if dynamic == 'nodelta':
			self.compDyn = lambda x: x
		elif dynamic == 'delta':
			self.compDyn = base.delta
		elif dynamic == 'ddelta':
			self.compDyn = base.ddelta
		else:
			raise Exception('unknown dynamic type')
			
		self.conf = conf
			
	## compute the features
	#
	#@param sig audio signal
	#@param rate sampling rate
	#
	#@return the features
	def __call__(self, sig, rate):
		
		if self.conf['snip_edges'] == 'True':
			#snip the edges
			sig = snip(sig, rate, float(self.conf['winlen']), float(self.conf['winstep']))
	
		#compute the features and energy
		feat, energy = self.compFeat(sig, rate, self.conf)
		
		#append the energy if requested
		if self.conf['include_energy'] == 'True':
			feat = np.append(feat,energy[:,np.newaxis],1)
			
		#add the dynamic information
		feat = self.compDyn(feat)
			
		return feat

		

##snip the edges of the utterance to fit the sliding window
#
#@param sig audio signal
#@param rate sampling rate
#@param winlen length of the sliding window [s]
#@param winstep stepsize of the sliding window [s]
#
#@return the snipped signal
def snip(sig, rate, winlen, winstep):
	# calculate the number of frames in the utterance as number of samples in the utterance / number of samples in the frame
	num_frames = int((len(sig)-winlen*rate)/(winstep*rate)) 
	# cut of the edges to fit the number of frames
	sig = sig[0:int(num_frames*winstep*rate + winlen*rate)]
	
	return sig
