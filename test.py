
#pylint : disable=E0401
from prepare.batch_dispenser import BatchDispenser
from prepare.feature_reader import FeatureReader
from prepare.batch_dispenser import UttTextDispenser
from prepare.batch_dispenser import PhonemeTextDispenser
import matplotlib.pyplot as plt
##for the moment i am assuming inputs are already computed and stored at
# locations specified in the input scripts.



aurora_path = "/esat/spchtemp/scratch/moritz/dataSets/aurora"
set_kind = "/train/40fbank"

train_feature_path = aurora_path + set_kind + "/" + "feats.scp"
train_cmvn_path = aurora_path + set_kind + "/" + "cmvn.scp"
utt2spk_path = aurora_path + set_kind + "/" + "utt2spk"
aurora_text_path = aurora_path + set_kind + "/" + "text"


feature_reader_aurora = FeatureReader(train_feature_path, train_cmvn_path,
                                    utt2spk_path)

MAX_TIME_AURORA = 2037
AURORA_LABELS = 32
BATCH_SIZE = 20

aurora_dispenser = UttTextDispenser(feature_reader_aurora, BATCH_SIZE,
                                   aurora_text_path, AURORA_LABELS,
                                   MAX_TIME_AURORA)
batched_data_aurora = aurora_dispenser.get_batch()

sparse_target_matrix = batched_data_aurora[1]
ix, val, shape = sparse_target_matrix
dense_target_matrix = BatchDispenser.sparse_to_dense(ix, val, shape)

# pylint: disable=C0325
# remove the python two print bracket warning...
print(dense_target_matrix[0, :])
print(aurora_dispenser.decode(dense_target_matrix[0, :]))

for i in range(1, 10):
    print(''.join(aurora_dispenser.decode(dense_target_matrix[i, :])))

ix, val, shape = batched_data_aurora[1]
dense_targets_aurora = BatchDispenser.sparse_to_dense(ix, val, shape)
plt.imshow(dense_targets_aurora)
plt.show()


if 0:
    timit_path = "/esat/spchtemp/scratch/moritz/dataSets/timit2"
    set_kind = "/train/40fbank"

    train_feature_path = timit_path + set_kind + "/" + "feats.scp"
    train_cmvn_path = timit_path + set_kind + "/" + "cmvn.scp"
    utt2spk_path = timit_path + set_kind + "/" + "utt2spk"
    timit_text_path = timit_path + set_kind + "/" + "text"

    feature_reader_timit = FeatureReader(train_feature_path, train_cmvn_path,
                                       utt2spk_path)

    MAX_TIME_TIMIT = 777
    TIMIT_LABELS = 39
    BATCH_SIZE = 462

    timit_dispenser = PhonemeTextDispenser(feature_reader_timit, BATCH_SIZE,
                                          timit_text_path, TIMIT_LABELS,
                                          MAX_TIME_TIMIT)

    batched_data_timit = timit_dispenser.get_batch()

    #take a look at the data
    plt.imshow(batched_data_timit[0][:, 0, :])
    plt.show()
    ix, val, shape = batched_data_timit[1]
    plt.imshow(BatchDispenser.sparse_to_dense(ix, val, shape))
    plt.show()
