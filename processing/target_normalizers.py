'''@file target_normalizers.py
Contains functions for target normalization, this is database and task dependent
'''

def aurora4_normalizer(transcription, alphabet):
    '''
    normalizer for Aurora 4 training transcriptions

    Args:
        transcription: the input transcription
        alphabet: the known characters alphabet

    Returns:
        the normalized transcription
    '''

    #create a dictionary of words that should be replaced
    replacements = {
        ',COMMA':'COMMA',
        '\"DOUBLE-QUOTE':'DOUBLE-QUOTE',
        '!EXCLAMATION-POINT':'EXCLAMATION-POINT',
        '&AMPERSAND':'AMPERSAND',
        '\'SINGLE-QUOTE':'SINGLE-QUOTE',
        '(LEFT-PAREN':'LEFT-PAREN',
        ')RIGHT-PAREN':'RIGHT-PAREN',
        '-DASH':'DASH',
        '-HYPHEN':'HYPHEN',
        '...ELLIPSIS':'ELLIPSIS',
        '.PERIOD':'PERIOD',
        '/SLASH':'SLASH',
        ':COLON':'COLON',
        ';SEMI-COLON':'SEMI-COLON',
        '<NOISE>': '',
        '?QUESTION-MARK': 'QUESTION-MARK',
        '{LEFT-BRACE': 'LEFT-BRACE',
        '}RIGHT-BRACE': 'RIGHT-BRACE'
        }

    #replace the words in the transcription
    replaced = ' '.join([word if word not in replacements
                         else replacements[word]
                         for word in transcription.split(' ')])

    #make the transcription lower case and put it into a list
    normalized = list(replaced.lower())

    #add the beginning and ending of sequence tokens
    normalized = ['<sos>'] + normalized + ['<eos>']

    #replace the spaces with <space>
    normalized = [character if character is not ' ' else '<space>'
                  for character in normalized]

    #replace unknown characters with <unk>
    normalized = [character if character in alphabet else '<unk>'
                  for character in normalized]

    return ' '.join(normalized)
