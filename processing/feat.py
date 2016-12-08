'''@file feat.py
Contains the class for feature computation'''

import base
import numpy as np

class FeatureComputer(object):
    '''A featurecomputer is used to compute a certain type of features'''

    def __init__(self, featureType, dynamic, conf):
        '''
        FeatureComputer constructor

        Args:
            featureType: string containing the type of features, optione are:
                fbank, mfcc and ssc.
            dynamic: the type of dynamic information added, options are:
                nodelta, delta and ddelta.
            conf: the feature configuration
        '''

        if featureType == 'fbank':
            self.comp_feat = base.logfbank
        elif featureType == 'mfcc':
            self.comp_feat = base.mfcc
        elif featureType == 'ssc':
            self.comp_feat = base.ssc
        else:
            raise Exception('unknown feature type')

        if dynamic == 'nodelta':
            self.comp_dyn = lambda x: x
        elif dynamic == 'delta':
            self.comp_dyn = base.delta
        elif dynamic == 'ddelta':
            self.comp_dyn = base.ddelta
        else:
            raise Exception('unknown dynamic type')

        self.conf = conf

    def __call__(self, sig, rate):
        '''
        compute the features

        Args:
            sig: audio signal
            rate: sampling rate

        Returns:
            the features
        '''

        if self.conf['snip_edges'] == 'True':
            #snip the edges
            sig = snip(sig, rate, float(self.conf['winlen']),
                       float(self.conf['winstep']))

        #compute the features and energy
        feat, energy = self.comp_feat(sig, rate, self.conf)

        #append the energy if requested
        if self.conf['include_energy'] == 'True':
            feat = np.append(feat, energy[:, np.newaxis], 1)

        #add the dynamic information
        feat = self.comp_dyn(feat)

        return feat

def snip(sig, rate, winlen, winstep):
    '''
    snip the edges of the utterance to fit the sliding window

    Args:
        sig: audio signal
        rate: sampling rate
        winlen: length of the sliding window [s]
        winstep: stepsize of the sliding window [s]

    Returns:
        the snipped signal
    '''
    # calculate the number of frames in the utterance as number of samples in
    #the utterance / number of samples in the frame
    num_frames = int((len(sig)-winlen*rate)/(winstep*rate))
    # cut of the edges to fit the number of frames
    sig = sig[0:int(num_frames*winstep*rate + winlen*rate)]

    return sig
