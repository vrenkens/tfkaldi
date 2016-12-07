'''@file target_coder.py
a file containing the target coders which can be used to encode and decode text,
alignments etc. '''

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

class TargetCoder(object):
    '''an abstract class for a target coder which can encode and decode target
    sequences'''

    __metaclass__ = ABCMeta

    def __init__(self, target_normalizer):
        '''
        TargetCoder constructor

        Args:
            target_normalizer: a target normalizer function
        '''

        #save the normalizer
        self.target_normalizer = target_normalizer

        #create an alphabet of possible targets
        alphabet = self.create_alphabet()

        #create a lookup dictionary for fast encoding
        self.lookup = OrderedDict([(character, index) for index, character
                                   in enumerate(alphabet)])

    @abstractmethod
    def create_alphabet(self):
        '''create the alphabet for the coder'''

    def encode(self, targets):
        '''
        encode a target sequence

        Args:
            targets: a string containing the target sequence

        Returns:
            A numpy array containing the encoded targets
        '''

        #normalize the targets
        normalized_targets = self.target_normalizer(targets, self.lookup.keys())

        encoded_targets = []

        for target in normalized_targets.split(' '):
            encoded_targets.append(self.lookup[target])

        return np.array(encoded_targets, dtype=np.uint32)

    def decode(self, encoded_targets):
        '''
        decode an encoded target sequence

        Args:
            encoded_targets: A numpy array containing the encoded targets

        Returns:
            A string containing the decoded target sequence
        '''

        targets = [self.lookup.keys()[encoded_target]
                   for encoded_target in encoded_targets]

        return ' '.join(targets)

    @property
    def num_labels(self):
        '''the number of possible labels'''

        return len(self.lookup)

class TextCoder(TargetCoder):
    '''a coder for text'''

    def create_alphabet(self):
        '''create the alphabet of characters'''

        alphabet = []
        # end of sentence token
        alphabet.append('<eos>')
        #start of sentence token
        alphabet.append('<sos>')
        #space
        alphabet.append('<space>')
        #comma
        alphabet.append(',')
        #period
        alphabet.append('.')
        #apostrophy
        alphabet.append('\'')
        #hyphen
        alphabet.append('-')
        #question mark
        alphabet.append('?')
        #unknown character
        alphabet.append('<unk>')
        #letters in the alphabet
        for letter in range(ord('a'), ord('z')+1):
            alphabet.append(chr(letter))

        return alphabet

class AlignmentCoder(TargetCoder):
    '''a coder for state alignments'''

    def __init__(self, target_normalizer, num_targets):
        '''
        AlignmentCoder constructor

        Args:
            target_normalizer: a target normalizer function
            num_targets: total number of targets
        '''

        self.num_targets = num_targets
        super(AlignmentCoder, self).__init__(target_normalizer)

    def create_alphabet(self):
        '''
        create the alphabet of alignment targets
        '''

        alphabet = [str(target) for target in range(self.num_targets)]

        return alphabet

class PhonemeEncoder(TargetCoder):
    """ Sets up a 39 element foldet phoneme alphabet."""

    def create_alphabet(self):
        """
        Create an alphabet of folded phonemes, according to
        "Speaker-Independent Phone Recognition Using Hidden Markov Models."
        """

        alphabet = ['sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
                    'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                    'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
                    't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
        return alphabet


class LasPhonemeEncoder(TargetCoder):
    """ Sets up a 39 element foldet phoneme alphabet."""

    def create_alphabet(self):
        """
        Create an alphabet of folded phonemes, according to
        "Speaker-Independent Phone Recognition Using Hidden Markov Models."
        It is very important to have the eos tokan at poistion one, because the las
        code assumes it is there!
        """

        alphabet = ['<eos>', '<sos>', 'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
                    'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                    'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
                    't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
        return alphabet
        