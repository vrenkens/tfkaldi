'''@file main.py
run this file to go through the neural net training procedure, look at the config files in the config directory to modify the settings'''

from __future__ import absolute_import, division, print_function


import os
from six.moves import configparser
#from neuralnetworks.nnet import Nnet
#from neuralnetworks.listen_net import Nnet
from neuralnetworks.las_net import Nnet
from processing import ark, prepare_data, feature_reader, batchdispenser, \
target_coder, target_normalizers, score

from shutil import copyfile
from IPython.core.debugger import Tracer; debug_here = Tracer();

#here you can set which steps should be executed.
#If a step has been executed in the past the result have been
#saved and the step does not have to be executed again
#(if nothing has changed)
TRAINFEATURES = False
TESTFEATURES = False
TRAIN = True
TEST_CTC = False
TEST_LAS = True

#read config file
config = configparser.ConfigParser()

#config_path = 'config/config_TIMIT.cfg'
#config_path = 'config/config_TIMIT_listener.cfg'
config_path = 'config/config_TIMIT_las.cfg'

#config.read()
config.read(config_path)
current_dir = os.getcwd()

#compute the features of the training set for DNN training
#if they are different then the GMM features.
if TRAINFEATURES:
    feat_cfg = dict(config.items('dnn-features'))

    print('------- computing DNN training features ----------')
    prepare_data.prepare_data(
        config.get('directories', 'train_data'),
        config.get('directories', 'train_features') + '/' + feat_cfg['name'],
        feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print('------- computing cmvn stats ----------')
    prepare_data.compute_cmvn(config.get('directories', 'train_features') + '/' + feat_cfg['name'])

# compute the features of the training set for DNN testing if
# they are different then the GMM features.
if TESTFEATURES:
    feat_cfg = dict(config.items('dnn-features'))

    print('------- computing DNN testing features ----------')
    prepare_data.prepare_data(
        config.get('directories', 'test_data'),
        config.get('directories', 'test_features') + '/' + feat_cfg['name'],
        feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print('------- computing cmvn stats ----------')
    prepare_data.compute_cmvn(config.get('directories', 'test_features') + '/' + feat_cfg['name'])

#get the feature input dim
reader = ark.ArkReader(
    config.get('directories', 'train_features') \
    + '/' + config.get('dnn-features', 'name') + '/feats.scp')
_, features, _ = reader.read_next_utt()
input_dim = features.shape[1]

#create the CTC coder
#coder = target_coder.PhonemeEncoder(target_normalizers.timit_phone_norm)

#create a Las coder
coder = target_coder.LasPhonemeEncoder(target_normalizers.timit_phone_norm_las)

#create the neural net
nnet = Nnet(config, input_dim, coder.num_labels)

if TRAIN:
    print('Backing up cfg for future reference...')
    copyfile(config_path,
         config.get('directories', 'expdir') + "/" \
         + config.get('nnet', 'name') + '/used_config.cfg')

    #only shuffle if we start with initialisation
    if config.get('nnet', 'starting_step') == '0':
        #shuffle the examples on disk
        print('------- shuffling examples ----------')
        prepare_data.shuffle_examples(
            config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name'))

    #create a feature reader
    featdir = config.get('directories', 'train_features') + '/' + \
      config.get('dnn-features', 'name')
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(
        featdir +'/feats_shuffled.scp', featdir + '/cmvn.scp',
        featdir + '/utt2spk', 0, max_input_length)
    #the path to the text file
    textfile = config.get('directories', 'train_data') + '/train39.text'
    #create a batch dispenser
    dispenser = batchdispenser.TextBatchDispenser(
        featreader, coder, int(config.get('nnet', 'batch_size')), textfile)
    #train the neural net
    print('------- training neural net ----------')
    nnet.train(dispenser)


if TEST_CTC:
    #use the neural net to calculate posteriors for the testing set.
    print('------- decoding test set ----------')
    savedir = config.get('directories', 'expdir') + '/' + config.get('nnet', 'name')
    decodedir = savedir + '/decode'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    featdir = config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')

    #create a feature reader
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(
        featdir + '/feats.scp', featdir + '/cmvn.scp',
        featdir + '/utt2spk', 0, max_input_length)

    #decode with the neural net
    resultsfolder = savedir + '/decode'
    nbests = nnet.decode(featreader, coder)

    #the path to the text file
    textfile = config.get('directories', 'test_data') + '/test39.text'

    #read all the reference transcriptions
    with open(textfile) as fid:
        lines = fid.readlines()

    references = dict()
    for line in lines:
        splitline = line.strip().split(' ')
        references[splitline[0]] = target_normalizers.timit_phone_norm(
            ' '.join(splitline[1:]), None)

    #compute the character error rate
    CER = score.CER(nbests, references)

    print('phoneme error rate: %f' % CER)
    print('Backing up cfg for future reference')
    copyfile(config_path,
             config.get('directories', 'expdir') + "/" \
             + config.get('nnet', 'name') + '/used_config.cfg')

if TEST_LAS:
    #use the neural net to calculate posteriors for the testing set.
    print('------- decoding test set ----------')
    savedir = config.get('directories', 'expdir') + '/' + config.get('nnet', 'name')
    decodedir = savedir + '/decode'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    featdir = config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')
    textfile = config.get('directories', 'test_data') + '/test39.text'

    #featdir = config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name')
    #textfile = config.get('directories', 'train_data') + '/train39.text'

    #create a feature reader
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(
        featdir + '/feats.scp', featdir + '/cmvn.scp',
        featdir + '/utt2spk', 0, max_input_length)

    #read all the reference transcriptions
    with open(textfile) as fid:
        lines = fid.readlines()

    references = dict()
    for line in lines:
        splitline = line.strip().split(' ')
        references[splitline[0]] = target_normalizers.timit_phone_norm(
            ' '.join(splitline[1:]), None)

    #decode with the neural net
    resultsfolder = savedir + '/decode'

    nbests = nnet.decode(featreader, coder)
    lev_dist = 0.0
    utts = 0.0
    for utt_id, decoded in nbests.iteritems():
        target = coder.encode(references[utt_id])
        utts += target.size
        lev_dist += score.edit_distance(target, decoded[0])

    lev_dist = lev_dist/utts

    print('set lev_dist: %f' % lev_dist)
    utt_id = references.keys()[0]
    print('Utterance_id', utt_id)
    print(references[utt_id])
    print(coder.decode(nbests[utt_id][0]))
    ex_lev = score.edit_distance(coder.encode(references[utt_id]),
                                 nbests[utt_id][0])
    target = coder.encode(references[utt_id])
    #print(target)
    #print(target.size)
    print('example lev_dist/length:', ex_lev/target.size)
    debug_here()