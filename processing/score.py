'''@file score.py
contains functions to score the system'''

import numpy as np
from IPython.core.debugger import Tracer; debug_here = Tracer();

def CER(nbests, references):
    '''
    compute the character error rate

    Args:
        nbests: the nbest lists, this is a dictionary with the uttreance id as
            key and a pair containing a list of hypothesis strings and
            a list of log probabilities
        references: the reference transcriptions as a list of strings

    Returns:
    the character error rate
    '''

    errors = 0
    num_labels = 0

    for utt in references:
        #get the single best decoded utterance as a list
        decoded = nbests[utt][0][0].split(' ')

        #get the reference as a list
        reference = references[utt].split(' ')

        #compute the error
        error_matrix = np.zeros([len(reference) + 1, len(decoded) + 1])

        error_matrix[:, 0] = np.arange(len(reference) + 1)
        error_matrix[0, :] = np.arange(len(decoded) + 1)

        for x in range(1, len(reference) + 1):
            for y in range(1, len(decoded) + 1):
                error_matrix[x, y] = min([
                    error_matrix[x-1, y] + 1, error_matrix[x, y-1] + 1,
                    error_matrix[x-1, y-1] + (reference[x-1] !=
                                              decoded[y-1])])

        errors += error_matrix[-1, -1]
        num_labels += len(reference)

    return errors/num_labels

def edit_distance(seq1, seq2):
    """ Calculate edit distance between sequences x and y using
        matrix dynamic programming.  Return distance.
        source, Ben Langmead:
        http://www.cs.jhu.edu/~langmea/resources/lecture_notes/dp_and_edit_dist.pdf
    """

    dmat = np.zeros((len(seq1)+1, len(seq2)+1), dtype=int)
    dmat[0, 1:] = range(1, len(seq2)+1)
    dmat[1:, 0] = range(1, len(seq1)+1)
    for i in range(1, len(seq1)+1):
        for j in range(1, len(seq2)+1):
            delt = 1 if seq1[i-1] != seq2[j-1] else 0
            dmat[i, j] = min(dmat[i-1, j-1]+delt, dmat[i-1, j]+1, dmat[i, j-1]+1)
    return dmat[len(seq1), len(seq2)]
