
#pylint : disable=E0401
from prepare.batch_dispenser import BatchDispenser
from prepare.feature_reader import FeatureReader
from prepare.batch_dispenser import UttTextDispenser
from prepare.batch_dispenser import PhonemeTextDispenser

import sys
sys.path.append('../timitNets')
from prepare_timit39 import load_batched_timit39

import matplotlib.pyplot as plt
import numpy as np
##for the moment i am assuming inputs are already computed stored at
# locations specified in the input scripts.


timit_path = "/esat/spchtemp/scratch/moritz/dataSets/timit2"
set_kind = "/train/40fbank"

train_feature_path = timit_path + set_kind + "/" + "feats.scp"
train_cmvn_path = timit_path + set_kind + "/" + "cmvn.scp"
utt2spk_path = timit_path + set_kind + "/" + "utt2spk"
timit_text_path = timit_path + set_kind + "/" + "text"

featureReaderTimit = FeatureReader(train_feature_path, train_cmvn_path,
                                   utt2spk_path)

MAX_TIME_TIMIT = 777
TIMIT_LABELS = 39
BATCH_SIZE = 462

timitDispenser = PhonemeTextDispenser(featureReaderTimit, BATCH_SIZE,
                                      timit_text_path, TIMIT_LABELS,
                                      MAX_TIME_TIMIT)

batched_data_timit = timitDispenser.get_batch()

#take a look at the data
plt.imshow(batched_data_timit[0][:,0,:])
plt.show()
ix, val, shape = batched_data_timit[1]
plt.imshow(BatchDispenser.sparse_to_dense(ix, val, shape))
plt.show()

print('Loading data')
data_batches = load_batched_timit39(timitDispenser.get_batch_count(), 1, 1)
batchedData, maxTimeSteps, totalN, batchSize = data_batches[0]

batched_data_timit_old = batchedData[0]

print("checking the sequence lengths:")
print(np.sum(np.abs(batched_data_timit[2] - batched_data_timit_old[2])))

print("checking input arrays:")
print(np.sum(np.abs(batched_data_timit_old[0] - batched_data_timit[0])))

ix, val, shape = batched_data_timit[1]
new_targets = BatchDispenser.sparse_to_dense(ix, val, shape)
ix_old, val_old, shape_old = batched_data_timit_old[1]
old_targets = BatchDispenser.sparse_to_dense(ix_old, val_old, shape_old)

print("checking target arrays:")
print(np.sum(np.abs(new_targets -  old_targets)))

print("checking sparse indices:")
print(np.sum(np.abs(ix - ix_old)))

plt.imshow(new_targets -  old_targets)
plt.show()



def decode(input_array):
    ''' Translate a target matrix row back in to phonemes. '''
    phone_map = {'aa': 0, 'ae': 1, 'ah': 2, 'aw': 3, 'ay': 4, 'b': 5,
                 'ch': 6, 'd': 7, 'dh': 8, 'dx': 9, 'eh': 10, 'er': 11,
                 'ey': 12, 'f': 13, 'g': 14, 'hh': 15, 'ih': 16, 'iy': 17,
                 'jh': 18, 'k': 19, 'l': 20, 'm': 21, 'n': 22, 'ng': 23,
                 'ow': 24, 'oy': 25, 'p': 26, 'r': 27, 's': 28, 'sh': 29,
                 'sil': 30, 't': 31, 'th': 32, 'uh': 33, 'uw': 34, 'v': 35,
                 'w': 36, 'y': 37, 'z': 38}
    #inverse the phone map.
    code_map = {code: phone for phone, code in phone_map.items()}
                 
    phones = []
    for i in range(0,len(input_array)):
        phones.append(code_map[int(input_array[i])])
    
    return phones
    
    
    
