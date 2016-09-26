##@package base
# Contains the functions that compute the features

#The MIT License (MIT)
#
#Copyright (c) 2013 James Lyons
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#the Software, and to permit persons to whom the Software is furnished to do so,
#subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
import numpy
import sigproc
from scipy.fftpack import dct
from scipy.ndimage import convolve1d

# make it python3.x compatible
try:
  xrange(1)
except:
  xrange=range

##Compute MFCC features from an audio signal.
#
#@param signal the audio signal from which to compute features. Should be an N*1 array
#@param samplerate the samplerate of the signal we are working with.
#@param conf feature configuration
#
#@return A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector, a numpy vector containing the signal log-energy
def mfcc(signal,samplerate,conf):
     
    feat,energy = fbank(signal,samplerate,conf)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:int(conf['numcep'])]
    feat = lifter(feat,float(conf['ceplifter']))
    return feat, numpy.log(energy)

##Compute fbank features from an audio signal.
#
#@param signal the audio signal from which to compute features. Should be an N*1 array
#@param samplerate the samplerate of the signal we are working with.
#@param conf feature configuration
#
#@return A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy vector containing the signal energy
def fbank(signal,samplerate,conf):
   
    highfreq= int(conf['highfreq']) 
    if highfreq < 0:
    	highfreq = samplerate/2
	
    signal = sigproc.preemphasis(signal,float(conf['preemph']))
    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate, float(conf['winstep'])*samplerate)
    pspec = sigproc.powspec(frames,int(conf['nfft']))
    energy = numpy.sum(pspec,1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log
        
    fb = get_filterbanks(int(conf['nfilt']),int(conf['nfft']),samplerate,int(conf['lowfreq']),highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log
    
    return feat,energy

##Compute log-fbank features from an audio signal.
#
#@param signal the audio signal from which to compute features. Should be an N*1 array
#@param samplerate the samplerate of the signal we are working with.
#@param conf feature configuration
#
#@return A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy vector containing the signal log-energy
def logfbank(signal,samplerate,conf):      
    feat,energy = fbank(signal,samplerate,conf)
    return numpy.log(feat), numpy.log(energy)

##Compute ssc features from an audio signal.
#
#@param signal the audio signal from which to compute features. Should be an N*1 array
#@param samplerate the samplerate of the signal we are working with.
#@param conf feature configuration
#
#@return A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy vector containing the signal log-energy
def ssc(signal,samplerate,conf):
        
    highfreq= int(conf['highfreq']) 
    if highfreq < 0:
    	highfreq = samplerate/2
    signal = sigproc.preemphasis(signal,float(conf['preemph']))
    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate, float(conf['winstep'])*samplerate)
    pspec = sigproc.powspec(frames,int(conf['nfft']))
    energy = numpy.sum(pspec,1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log
        
    fb = get_filterbanks(int(conf['nfilt']),int(conf['nfft']),samplerate,int(conf['lowfreq']),highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    R = numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(pspec,1)),(numpy.size(pspec,0),1))
    
    return numpy.dot(pspec*R,fb.T) / feat, numpy.log(energy)
    
##Convert a value in Hertz to Mels
#
#@param hz a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
#
#@return a value in Mels. If an array was passed in, an identical sized array is returned.
def hz2mel(hz):
    return 2595 * numpy.log10(1+hz/700.0)
    
##Convert a value in Mels to Hertz
#
#@param mel a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
#
#@return a value in Hertz. If an array was passed in, an identical sized array is returned.
def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

##Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
#
#@param nfilt the number of filters in the filterbank, default 20.
#@param nfft the FFT size. Default is 512.
#@param samplerate the samplerate of the signal we are working with. Affects mel spacing.
#@param lowfreq lowest band edge of mel filters, default 0 Hz
#@param highfreq highest band edge of mel filters, default samplerate/2
#
#@return A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):

    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft/2+1])
    for j in xrange(0,nfilt):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank                 
    
##Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the magnitude of the high frequency DCT coeffs.
#
#@param cepstra the matrix of mel-cepstra, will be numframes * numcep in size.
#@param L the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
#
#@return the lifted cepstra
def lifter(cepstra,L=22):
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1+ (L/2)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra
        
##Compute the first order derivative of the features
#
#@param features the input features
#
#@return the firs order derivative
def deriv(features):
	return convolve1d(features, [2,1,0,-1,-2], 0)
	
##concatenate the first order derivative to the features
#
#@param features the input features
#
#@return the features concatenated with the first order derivative
def delta(features):
	return numpy.concatenate((features, deriv(features)), 1)
	
##concatenate the first and second order derivative to the features
#
#@param features the input features
#
#@return the features concatenated with the first and second order derivative
def ddelta(features):
	deltafeat = deriv(features)
	return numpy.concatenate((features, deltafeat, deriv(deltafeat)), 1)

    
