'''
#@package batchdispenser
# contain the functionality for read features and batches
# of features for neural network training and testing
'''

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import prepare.ark

## Class that dispenses batches of data for mini-batch training
class BatchDispenser(metaclass=ABCMeta):
    '''
    BatchDispenser interface cannot be created but gives methods to its
    child classes.
    '''

    @abstractmethod
    def normalize_targets(self, target_list):
        '''abstract normalize targets function. Must be overwritten
           for different data types, i.e. phonemes or letters. Here
           phonemes can be folded or unwanted letters removed.'''
        pass

    @abstractmethod
    def encode(self, char_lst):
        '''This method defines the the way the targets are encoded.
        It must be overwritten in every child class.'''
        pass

    def __init__(self, featureReader, size, text_path, num_labels, max_time):
        '''Abstract constructor for nonexisting general data sets.'''
        #store the feature reader
        # pylint: disable=C0103
        self.featureReader = featureReader
        #save the number of labels
        self.num_labels = num_labels
        text_file = open(text_path)
        self.text_lines = text_file.read().splitlines()

        #get a dictionary connecting training utterances and transcriptions.
        self.text_dict = {}
        for i in range(0, len(self.text_lines)):
            tmp = self.text_lines[i].split(' ')
            self.text_dict.update({tmp[0]: self.normalize_targets(tmp[1:])})

        #store the batch size
        self.size = size

        #store the max number of time steps (None if unkown)
        self.max_time = max_time



    def get_batch(self):
        '''
        get a batch of features and alignments in one-hot encoding

        @return a batch of data, the corresponding labels in one hot encoding
        '''

        #set up the data lists.
        input_list = []
        target_list = []
        elmnt_cnt = 0

        while elmnt_cnt < self.size:
            #read utterance
            utt_id, utt_mat, _ = self.featureReader.get_utt()

            #get transcription
            transcription = self.text_dict[utt_id]
            encoded_target = self.encode(transcription)

            input_list.append(utt_mat.transpose())
            target_list.append(encoded_target)

            elmnt_cnt += 1

        #do the input padding, convert the targets
        #into sparse one hot-matrices.
        batch_inputs, sparse_target_data, max_steps = \
            self.data_lists_to_batch(input_list, target_list)

        return batch_inputs, sparse_target_data, max_steps

    def split(self):
        '''
        split of the part that has allready been read by the batchdispenser,
        this can be used to read a validation set and
        then split it of from the rest
        '''
        self.featureReader.split()

    def skip_batch(self):
        '''skip a batch'''
        elmnt_cnt = 0
        while elmnt_cnt < self.size:
            #read utterance
            _ = self.featureReader.nextId()
            #update number of utterances in the batch
            elmnt_cnt += 1

    def return_batch(self):
        '''Reset to previous batch'''
        elmnt_cnt = 0
        while elmnt_cnt < self.size:
            #read utterance
            _ = self.featureReader.prevId()
            #update number of utterances in the batch
            elmnt_cnt += 1

    def get_batch_size(self):
        '''Returns the number of utterances per batch.'''
        return self.size

    def get_batch_count(self):
        '''Return the number of batches with the given size can be dispensed
        until looping over the data begins.'''
        return int(self.get_num_utt()/self.size)


    def get_num_utt(self):
        '''@return the number of utterances
            the current instance can dinspense.'''
        return len(self.text_dict)

    @staticmethod
    def target_list_to_sparse_tensor(target_list):
        '''make tensorflow SparseTensor from list of targets, with each element
           in the list being a list or array with the values of the target sequence
           (e.g., the integer values of a character map for an ASR target string)
        '''
        indices = []
        vals = []
        lengths = []
        for t_i, target in enumerate(target_list):
            for seq_i, val in enumerate(target):
                lengths.append(len(target))
                if val != 0:
                    indices.append([t_i, seq_i])
                    vals.append(val)
        shape = [len(target_list), np.max(lengths)]

        return (np.array(indices), np.array(vals), np.array(shape))

    def data_lists_to_batch(self, input_list, target_list):
        '''Takes a list of input matrices and a list of target arrays and returns
           a batch, which is 3-element tuple of inputs,
           targets, and sequence lengths.
           inputs:
                input_list: list of 2-d numpy arrays with dimensions n_features x timesteps
                target_list: list of 1-d arrays or lists of ints
           returns: data_batch:
                    consists of:
                        inputs  = 3-d array w/ shape nTimeSteps x batch_size x n_features
                        targets = tuple required as input for SparseTensor
                        seqLengths = 1-d array with int number of timesteps for
                                     each sample in batch
            '''

        assert len(input_list) == len(target_list)
        n_features = input_list[0].shape[0]

        if self.max_time is None:
            max_steps = 0
            for inp in input_list:
                max_steps = max(max_steps, inp.shape[1])
        else:
            max_steps = self.max_time

        #randIxs = np.random.permutation(len(input_list)) #randomly mix the batches.
        batch_ixs = np.asarray(range(len(input_list))) #do not randomly permutate.
        start, end = (0, self.size)

        batch_seq_lengths = np.zeros(self.size)
        for batch_i, orig_i in enumerate(batch_ixs[start:end]):
            batch_seq_lengths[batch_i] = input_list[orig_i].shape[-1]
        batch_inputs = np.zeros((max_steps, self.size, n_features))
        batch_target_list = []
        for batch_i, orig_i in enumerate(batch_ixs[start:end]):
            pad_secs = max_steps - input_list[orig_i].shape[1]
            batch_inputs[:, batch_i, :] = \
                np.pad(input_list[orig_i].transpose(), ((0, pad_secs), (0, 0)),
                       'constant', constant_values=0)
            batch_target_list.append(target_list[orig_i])

        sparse_target_data = BatchDispenser.target_list_to_sparse_tensor(
            batch_target_list)

        return batch_inputs, sparse_target_data, batch_seq_lengths

    @staticmethod
    def sparse_to_dense(indices, values, shape):
        '''
        Convert a tensorflow style sparse matrix into a dense numpy array.
        '''
        dense_array = np.zeros(shape)

        for i, index in enumerate(indices):
            dense_array[index[0], index[1]] = values[i]
        return dense_array

    @staticmethod
    def dense_to_sparse(dense_matrix):
        '''
        Convert a dese numpy array into a tensorflow style sparse matrix.
        '''
        shape = dense_matrix.shape
        indices = []
        values = []
        for row in range(0, shape[0]):
            for col in range(0, shape[1]):
                if dense_matrix[row, col] != 0:
                    indices.append([row, col])
                    values.append(dense_matrix[row, col])
        return np.array(indices), np.array(values), np.array(shape)

    @staticmethod
    def array_list_to_dense(array_list, shape):
        '''
        Merge a list of arrays row wise into a large dense matrix.
        '''
        dense_array = np.zeros(shape)
        for i, array in enumerate(array_list):
            dense_array[i, 0:len(array)] = array
        return dense_array



## Class that dispenses batches of data for mini-batch training
class UttTextDispenser(BatchDispenser):
    '''
    Defines a batch dispenser, which uses text targets.
    '''
    def __init__(self, featureReader, size, text_path, num_labels, max_time):
        #generate a list of allowed unicodes. Everything else will be
        #replaced with (?)
        allowed_chrs = []
        allowed_chrs.append(' ')
        allowed_chrs.append('(')
        allowed_chrs.append(')')
        allowed_chrs.append(',')
        allowed_chrs.append('.')
        allowed_chrs.append('<')
        allowed_chrs.append('>')
        allowed_chrs.append('?')
        for counti in range(ord('a'), ord('z')):
            allowed_chrs.append(chr(counti))
        self.allowed_chrs = allowed_chrs

        code_dict = {}
        for key, char in enumerate(allowed_chrs):
            code_dict.update({char: key})
        self.code_dict = code_dict

        super().__init__(featureReader, size, text_path,
                         num_labels, max_time)

    def encode(self, char_lst):
        '''
        Encode a character using the las encoding specified in self.code
        dict
        '''
        encoded = []
        for char in char_lst:
            encoded.append(self.code_dict[char])
        return np.array(encoded, dtype=np.uint8)


    def normalize_targets(self, target_list):
        '''
        Normalize the input word list, assuming a character level model,
        with LAS character encoding:
            see: Chan et el - 2015 Listen Attend and Spell.
        @param target_list list of uppercase words.
        '''
        tmp_lst = []
        tmp_lst.append('<')
        for word in target_list:
            if word == ",COMMA":
                tmp_lst.append(',')
            elif word == '.PERIOD':
                tmp_lst.append('.')
            else:
                tmp_lst.append(' ')
                tmp_lst.append(word)
        tmp_lst.append('>')

        #check the string for unknowns:
        norm_lst = []
        for word in tmp_lst:
            if word == '(sos)' or word == '(eos)':
                norm_lst.append(word)
            else:
                for any_chr in word:
                    lower_c = any_chr.lower()
                    if lower_c not in self.allowed_chrs:
                        norm_lst.append('?')
                    else:
                        norm_lst.append(lower_c)
        return norm_lst

class PhonemeTextDispenser(BatchDispenser):
    '''Defines a batch dispenser wich uses phoneme targets'''

    def __init__(self, featureReader, size, text_path, num_labels, max_time):
        #initialize the member variables.
        super().__init__(featureReader, size, text_path,
                         num_labels, max_time)

        #check the vocabulary.
        vocab_dict = {}
        for i in range(0, len(self.text_lines)):
            tmp = self.text_lines[i].split(' ')
            for wrd in tmp[1:]:
                if wrd in vocab_dict:
                    vocab_dict[wrd] += 1
                else:
                    #print('adding: ' + str({wrd: 0}))
                    vocab_dict.update({wrd: 0})

        phones = list(vocab_dict.keys())
        phones.sort()

        phone_map = {}
        for no, phoneme in enumerate(phones):
            phone_map.update({phoneme: no})
        self.phone_map = phone_map
        
        #check if the phoneme number in the data matches with what the user
        #thinks it is.
        assert(num_labels == len(self.phone_map))

    def normalize_targets(self, target_list):
        '''Phoneme folding could be done here. It is however already done
           in kaldi for the timit data set version in use when this code was
           written.'''
        return target_list

    def encode(self, phone_lst):
        '''Encode a list of phonemes using the phoneme dict set up during
           initialization.'''

        targets = []
        for phone in phone_lst:
            targets.append(self.phone_map[phone])
        return np.array(targets, dtype=np.uint8)
        
    
