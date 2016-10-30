'''@file feature_reader.py
reading features and applying cmvn and splicing them'''

from copy import copy
import numpy as np
import processing.ark as ark
import processing.readfiles as readfiles

class FeatureReader(object):
    '''Class that can read features from a Kaldi archive and process
    them (cmvn and splicing)'''

    def __init__(self, scp_path, cmvn_path, utt2spk_path,
                 context_width, max_input_length):
        '''
        create a FeatureReader object

        Args:
            scp_path: path to the features .scp file
            cmvn_path: path to the cmvn file
            utt2spk_path:path to the file containing the mapping from utterance
                ID to speaker ID
            context_width: context width for splicing the features
            max_input_length: the maximum length of all the utterances in the
                scp file
        '''

        #create the feature reader
        self.reader = ark.ArkReader(scp_path)

        #create a reader for the cmvn statistics
        self.reader_cmvn = ark.ArkReader(cmvn_path)

        #save the utterance to speaker mapping
        self.utt2spk = readfiles.read_utt2spk(utt2spk_path)

        #store the context width
        self.context_width = context_width

        #store the max length
        #TODO: Question for vincent, why is this here? The object does not
        #      use this member variable.
        self.max_input_length = max_input_length

    def get_utt(self):
        '''
        read the next features from the archive, normalize and splice them

        Returns:
            the normalized and spliced features
        '''

        #read utterance
        (utt_id, utt_mat, looped) = self.reader.read_next_utt()

        #apply cmvn
        cmvn_stats = self.reader_cmvn.read_utt(self.utt2spk[utt_id])
        utt_mat = apply_cmvn(utt_mat, cmvn_stats)

        if self.context_width is not 0:
            #splice the utterance
            utt_mat = splice(utt_mat, self.context_width)

        return utt_id, utt_mat, looped

    def next_id(self):
        '''
        only gets the ID of the next utterance

        moves forward in the reader

        Returns:
            the ID of the uterance
        '''

        return self.reader.read_next_scp()

    def prev_id(self):
        '''
        only gets the ID of the previous utterance

        moves backward in the reader

        Returns:
            the ID of the uterance
        '''

        return self.reader.read_previous_scp()

    def split(self):
        '''split of the features that have been read so far'''
        self.reader.split()

    def split_utt(self, num_utt):
        '''Remove num_utt utterances from the feature reader and
            place them in a new reader object.'''
        new_feature_reader = copy(self)
        new_feature_reader.reader = self.reader.split_utt(num_utt)
        return new_feature_reader

    def get_utt_no(self):
        '''Return the total number of utterances, which could possible be
        loaded from the associated data file. '''
        return len(self.reader.scp_data)


def apply_cmvn(utt, stats):
    '''
    apply mean and variance normalisation

    The mean and variance statistics are computed on previously seen data

    Args:
        utt: the utterance feature numpy matrix
        stats: a numpy array containing the mean and variance statistics. The
            first row contains the sum of all the fautures and as a last element
            the total number of features. The second row contains the squared
            sum of the features and a zero at the end

    Returns:
        a numpy array containing the mean and variance normalized features
    '''

    #compute mean
    mean = stats[0, :-1]/stats[0, -1]

    #compute variance
    variance = stats[1, :-1]/stats[0, -1] - np.square(mean)

    #return mean and variance normalised utterance
    return np.divide(np.subtract(utt, mean), np.sqrt(variance))

def splice(utt, context_width):
    '''
    splice the utterance

    Args:
        utt: numpy matrix containing the utterance features to be spliced
        context_width: how many frames to the left and right should
            be concatenated

    Returns:
        a numpy array containing the spliced features
    '''

    #create spliced utterance holder
    utt_spliced = np.zeros(
        shape=[utt.shape[0], utt.shape[1]*(1+2*context_width)],
        dtype=np.float32)

    #middle part is just the uttarnce
    utt_spliced[:, context_width*utt.shape[1]:
                (context_width+1)*utt.shape[1]] = utt

    for i in range(context_width):

        #add left context
        utt_spliced[i+1:utt_spliced.shape[0],
                    (context_width-i-1)*utt.shape[1]:
                    (context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1, :]

         #add right context
        utt_spliced[0:utt_spliced.shape[0]-i-1,
                    (context_width+i+1)*utt.shape[1]:
                    (context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0], :]

    return utt_spliced
