'''
@file ark.py
contains the .ark io functionality

Copyright 2014    Yajie Miao    Carnegie Mellon University
           2015    Yun Wang      Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.
See the Apache 2 License for the specific language governing permissions and
limitations under the License.
'''

import struct
import numpy as np

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)

class ArkReader(object):
    '''
    Class to read Kaldi ark format. Each time, it reads one line of the .scp
    file and reads in the corresponding features into a numpy matrix. It only
    supports binary-formatted .ark files. Text and compressed .ark files are not
    supported. The inspiration for this class came from pdnn toolkit (see
    licence at the top of this file) (https://github.com/yajiemiao/pdnn)
    '''

    def __init__(self, scp_path):
        '''
        ArkReader constructor

        Args:
            scp_path: path to the .scp file
        '''

        self.scp_position = 0
        fin = open(scp_path, "r")
        self.utt_ids = []
        self.scp_data = []
        line = fin.readline()
        while line != '' and line != None:
            utt_id, path_pos = line.replace('\n', '').split(' ')
            path, pos = path_pos.split(':')
            self.utt_ids.append(utt_id)
            self.scp_data.append((path, pos))
            line = fin.readline()

        fin.close()

    def read_utt_data(self, index):
        '''
        read data from the archive

        Args:
            index: index of the utterance that will be read

        Returns:
            a numpy array containing the data from the utterance
        '''

        ark_read_buffer = open(self.scp_data[index][0], 'rb')
        ark_read_buffer.seek(int(self.scp_data[index][1]), 0)
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != "B":
            print "Input .ark file is not binary"
            exit(1)
        if header[1] == "C":
            print "Input .ark file is compressed"
            exit(1)

        _, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        _, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        if header[1] == "F":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4),
                                    dtype=np.float32)
        elif header[1] == "D":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8),
                                    dtype=np.float64)

        utt_mat = np.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_mat

    def read_next_utt(self):
        '''
        read the next utterance in the scp file

        Returns:
            the utterance ID of the utterance that was read, the utterance data,
            bool that is true if the reader looped back to the beginning
        '''

        if len(self.scp_data) == 0:
            return None, None, True

        #if at end of file loop around
        if self.scp_position >= len(self.scp_data):
            looped = True
            self.scp_position = 0
        else:
            looped = False

        self.scp_position += 1

        return (self.utt_ids[self.scp_position-1],
                self.read_utt_data(self.scp_position-1), looped)

    def read_next_scp(self):
        '''
        read the next utterance ID but don't read the data

        Returns:
            the utterance ID of the utterance that was read
        '''

        #if at end of file loop around
        if self.scp_position >= len(self.scp_data):
            self.scp_position = 0

        self.scp_position += 1

        return self.utt_ids[self.scp_position-1]

    def read_previous_scp(self):
        '''
        read the previous utterance ID but don't read the data

        Returns:
            the utterance ID of the utterance that was read
        '''

        if self.scp_position < 0: #if at beginning of file loop around
            self.scp_position = len(self.scp_data) - 1

        self.scp_position -= 1

        return self.utt_ids[self.scp_position+1]

    def read_utt(self, utt_id):
        '''
        read the data of a certain utterance ID

        Returns:
            the utterance data corresponding to the ID
        '''

        return self.read_utt_data(self.utt_ids.index(utt_id))

    def split(self):
        '''Split of the data that was read so far'''

        self.scp_data = self.scp_data[self.scp_position:-1]
        self.utt_ids = self.utt_ids[self.scp_position:-1]

class ArkWriter(object):
    '''
    Class to write numpy matrices into Kaldi .ark file and create the
    corresponding .scp file. It only supports binary-formatted .ark files. Text
    and compressed .ark files are not supported. The inspiration for this class
    came from pdnn toolkit (see licence at the top of this file)
    (https://github.com/yajiemiao/pdnn)
    '''

    def __init__(self, scp_path, default_ark):
        '''
        Arkwriter constructor

        Args:
            scp_path: path to the .scp file that will be written
            default_ark: the name of the default ark file (used when not
                specified)
        '''

        self.scp_path = scp_path
        self.scp_file_write = open(self.scp_path, 'w')
        self.default_ark = default_ark

    def write_next_utt(self, utt_id, utt_mat, ark_path=None):
        '''
        read an utterance to the archive

        Args:
            ark_path: path to the .ark file that will be used for writing
            utt_id: the utterance ID
            utt_mat: a numpy array containing the utterance data
        '''

        ark = ark_path or self.default_ark
        ark_file_write = open(ark, 'ab')
        utt_mat = np.asarray(utt_mat, dtype=np.float32)
        rows, cols = utt_mat.shape
        ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        pos = ark_file_write.tell()
        ark_file_write.write(struct.pack('<xcccc', 'B', 'F', 'M', ' '))
        ark_file_write.write(struct.pack('<bi', 4, rows))
        ark_file_write.write(struct.pack('<bi', 4, cols))
        ark_file_write.write(utt_mat)
        self.scp_file_write.write('%s %s:%s\n' % (utt_id, ark, pos))
        ark_file_write.close()

    def close(self):
        '''close the ark writer'''

        self.scp_file_write.close()
