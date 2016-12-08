'''@file nnet.py
contains the functionality for a Kaldi style neural network'''
from __future__ import absolute_import, division, print_function

import shutil
import os
import itertools
import pickle
import tensorflow as tf

from neuralnetworks.classifiers.las_model import LasModel
from neuralnetworks.classifiers.las_model import GeneralSettings
from neuralnetworks.classifiers.las_model import ListenerSettings
from neuralnetworks.classifiers.las_model import AttendAndSpellSettings
from neuralnetworks.reg_trainer import LasCrossEnthropyTrainer
from neuralnetworks.decoder import SimpleSeqDecoder

from IPython.core.debugger import Tracer; debug_here = Tracer();

class Nnet(object):
    '''a class for using a DBLTSM with CTC for ASR'''

    def __init__(self, conf, input_dim, num_labels):
        '''
        Nnet constructor

        Args:
            conf: nnet configuration
            input_dim: network input dimension
            num_labels: number of target labels
        '''

        #get nnet structure configs
        self.net_conf = dict(conf.items('nnet'))
        self.feat_conf = dict(conf.items('dnn-features'))

        #define location to save neural nets
        self.net_conf['savedir'] = (conf.get('directories', 'expdir')
                                    + '/' + self.net_conf['name'])

        if not os.path.isdir(self.net_conf['savedir'] + '/training'):
            os.makedirs(self.net_conf['savedir'] + '/training')
        if not os.path.isdir(self.net_conf['savedir'] + '/validation'):
            os.makedirs(self.net_conf['savedir'] + '/validation')

        #save the input dim
        self.input_dim = input_dim

        #create a Listener model, which will be paired with CTC later.
        #"mel_feature_no, batch_size, target_label_no, dtype"
        self.gset = GeneralSettings(int(self.feat_conf['nfilt']),
                                    int(self.net_conf['numutterances_per_minibatch']),
                                    int(num_labels), tf.float32)
        #lstm_dim, plstm_layer_no, output_dim, out_weights_std
        self.lset = ListenerSettings(int(self.net_conf['num_units']),
                                     int(self.net_conf['num_layers']),
                                     None, None)
        #decoder_state_size, feedforward_hidden_units, feedforward_hidden_layers
        self.asset = AttendAndSpellSettings(self.net_conf['state_size'],
                                            self.net_conf['net_size'],
                                            self.net_conf['n_hidden'],
                                            self.net_conf['net_out_prob'])

        self.classifier = LasModel(self.gset, self.lset, self.asset)

        #create the wavenet
        #self.classifier = Wavenet(num_labels + 1, int(self.net_conf['num_layers']),
        #                          2, int(self.net_conf['num_units']), 7)


    def train(self, dispenser):
        '''
        Train the neural network

        Args:
            dispenser: a batchdispenser for training
        '''

        #get the validation set
        if int(self.net_conf['valid_batches']) > 0:
            val_data, val_labels = zip(
                *[dispenser.get_batch()
                  for _ in range(int(self.net_conf['valid_batches']))])

            val_data = list(itertools.chain.from_iterable(val_data))
            val_labels = list(itertools.chain.from_iterable(val_labels))
        else:
            val_data = None
            val_labels = None

        dispenser.split()

        #compute the total number of steps
        num_steps = int(dispenser.num_batches *int(self.net_conf['num_epochs']))

        #set the step to the starting step
        step = int(self.net_conf['starting_step'])


        #go to the point in the database where the training was at checkpoint
        for _ in range(step):
            dispenser.skip_batch()

        if self.net_conf['numutterances_per_minibatch'] == '-1':
            numutterances_per_minibatch = dispenser.size
        else:
            numutterances_per_minibatch = int(
                self.net_conf['numutterances_per_minibatch'])

        #put the las in a cross entropy training environment
        print('building the training graph')
        trainer = LasCrossEnthropyTrainer(
            self.classifier, self.input_dim, dispenser.max_input_length,
            dispenser.max_target_length,
            float(self.net_conf['initial_learning_rate']),
            float(self.net_conf['learning_rate_decay']),
            num_steps, numutterances_per_minibatch)

        #start the visualization if it is requested
        if self.net_conf['visualise'] == 'True':
            if os.path.isdir(self.net_conf['savedir'] + '/logdir'):
                shutil.rmtree(self.net_conf['savedir'] + '/logdir')

            trainer.start_visualization(self.net_conf['savedir'] + '/logdir')


        #create lists to store error and loss.
        loss_lst = []
        val_loss_lst = []
        val_error_list = []

        #start a tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        with tf.Session(graph=trainer.graph, config=config):
            #initialise the trainer
            trainer.initialize()
            
            #load the neural net if the starting step is not 0
            if step > 0:
                trainer.restore_trainer(self.net_conf['savedir']
                                        + '/training/step' + str(step))

            #do a validation step
            if val_data is not None:
                validation_error, val_loss = trainer.evaluate(val_data, val_labels)
                print('validation error at step %d: %f' % (step, validation_error))
                print('validation loss at step %d: %f' % (step, val_loss))
                validation_step = step
                trainer.save_trainer(self.net_conf['savedir']
                                     + '/validation/validated')
                num_retries = 0
                val_error_list.append([step, validation_error])
                val_loss_lst.append([step, val_loss])

            #start the training iteration
            while step < num_steps:

                #get a batch of data
                batch_data, batch_labels = dispenser.get_batch()

                #update the model
                loss, lr = trainer.update(batch_data, batch_labels)
                loss_lst.append([step, loss])

                #print the progress
                print ('step %d/%d loss: %f, learning rate: %f'
                       %(step, num_steps, loss, lr))

                #increment the step
                step += 1

                #validate the model if required
                if (step%int(self.net_conf['valid_frequency']) == 0
                        and val_data is not None):

                    current_error, val_loss = trainer.evaluate(val_data, val_labels)
                    print('validation error at step %d: %f' %(step, current_error))
                    print('validation loss at step %d: %f' %(step, val_loss))
                    val_error_list.append([step, current_error])
                    val_loss_lst.append([step, val_loss])

                    if self.net_conf['valid_adapt'] == 'True':
                        #if the loss increased, half the learning rate and go
                        #back to the previous validation step
                        if current_error > validation_error:

                            #go back in the dispenser
                            for _ in range(step-validation_step):
                                dispenser.return_batch()

                            #load the validated model
                            trainer.restore_trainer(self.net_conf['savedir']
                                                    + '/validation/validated')

                            #halve the learning rate
                            trainer.halve_learning_rate()

                            #save the model to store the new learning rate
                            trainer.save_trainer(self.net_conf['savedir']
                                                 + '/validation/validated')

                            step = validation_step

                            if num_retries == int(self.net_conf['valid_retries']):
                                print('''the validation loss is worse,
                                         terminating training''')
                                break

                            print('''the validation loss is worse, returning to
                                     the previously validated model with halved
                                     learning rate''')

                            num_retries += 1

                            continue

                        else:
                            validation_error = current_error
                            validation_step = step
                            num_retries = 0
                            trainer.save_trainer(self.net_conf['savedir']
                                                 + '/validation/validated')

                #save the model if at checkpoint
                if step%int(self.net_conf['check_freq']) == 0:
                    trainer.save_trainer(self.net_conf['savedir'] + '/training/step'
                                         + str(step))

            #save the final model
            trainer.save_model(self.net_conf['savedir'] + '/final')
            pickle.dump([loss_lst, val_loss_lst, val_error_list],
                        open(self.net_conf['savedir']+ "/" \
                             + 'plot.pkl', "wb"))

    def decode(self, reader, target_coder):
        '''
        compute pseudo likelihoods the testing set

        Args:
            reader: a feature reader object to read features to decode
            target_coder: target coder object to decode the target sequences

        Returns:
            a dictionary with the utterance id as key and a pair as Value
            containing:
                - a list of hypothesis strings
                - a numpy array of log probabilities for the hypotheses
        '''

        #create a decoder
        print('building the decoding graph')
        #mel_feature_no, batch_size, target_label_no, dtype
        gset = GeneralSettings(self.gset.mel_feature_no,
                               1,
                               self.gset.target_label_no,
                               self.gset.dtype)
        decoding_classifier = \
            LasModel(gset, self.lset, self.asset, decoding=True)
        decoder = SimpleSeqDecoder(decoding_classifier, self.input_dim,
                                   reader.max_input_length)
        #start tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101

        nbests = dict()

        with tf.Session(graph=decoder.graph, config=config):

            #load the model
            decoder.restore(self.net_conf['savedir'] + '/final')

            #feed the utterances one by one to the neural net
            while True:
            #if 1:
                #for _ in range(10):
                utt_id, utt_mat, looped = reader.get_utt()

                if looped:
                    break

                #compute predictions
                encoded_hypotheses = decoder(utt_mat)
                nbests[utt_id] = encoded_hypotheses

        return nbests
