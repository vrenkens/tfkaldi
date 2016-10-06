import numpy as np

import prepare.ark as ark
import prepare.kaldiInterface as kaldiInterface


def apply_cmvn(utt, stats):
    '''apply mean and variance normalisation based on the previously computed statistics

    @param utt the utterance feature numpy matrix
    @param stats a numpy array containing the mean and variance statistics.
    The first row contains the sum of all the fautures and as a last
    element the total numbe of features.  The second row contains the squared
    sum of the features and a zero at the end.
    @return a numpy array containing the mean and variance normalized features
    '''

    mean = stats[0, :-1]/stats[0, -1]
    variance = stats[1, :-1]/stats[0, -1] - np.square(mean)

    #return mean and variance normalised utterance
    return np.divide(np.subtract(utt, mean), np.sqrt(variance))


## Class that can read features from a Kaldi archive and process them
#  (cmvn and splicing)
class FeatureReader:
    '''create a FeatureReader object

    @param scp_path: path to the features .scp file
    @param cmvn_path: path to the cmvn file
    @param utt2spk_path:path to the file containing the mapping
            from utterance ID to speaker ID
    @param target_path: file system path to the target text transcriptions.
    '''
    def __init__(self, scp_path, cmvn_path, utt2spk_path):
        #create the feature reader
        self.reader = ark.ArkReader(scp_path)

        #create a reader for the cmvn statistics
        self.reader_cmvn = ark.ArkReader(cmvn_path)

        #save the utterance to speaker mapping
        self.utt2spk = kaldiInterface.read_utt2spk(utt2spk_path)

    def get_utt(self):
        '''
        read the next features from the archive and normalize them
        @return the normalized features
        '''
        #read utterance
        (utt_id, utt_mat, looped) = self.reader.read_next_utt()

        #apply cmvn
        cmvn_stats = self.reader_cmvn.read_utt(self.utt2spk[utt_id])
        utt_mat = apply_cmvn(utt_mat, cmvn_stats)

        return utt_id, utt_mat, looped

    def next_id(self):
        '''
        only gets the ID of the next utterance
        (also moves forward in the reader)

        @return the ID of the uterance
        '''
        return self.reader.read_next_scp()

    def prev_id(self):
        '''
        only gets the ID of the previous utterance
        #(also moves backward in the reader)

        @return the ID of the uterance
        '''
        return self.reader.read_previous_scp()

    def split(self):
        ''' split of the features that have been read so far'''
        self.reader.split()
