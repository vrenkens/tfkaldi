'''@file nnet.py
contains the functionality for a Kaldi style neural network'''

import shutil
import os
import itertools
import numpy as np
import tensorflow as tf
import classifiers.activation
from classifiers.dnn import DNN
from trainer import CrossEnthropyTrainer
from decoder import Decoder

class Nnet(object):
    '''a class for a neural network that can be used together with Kaldi'''

    def __init__(self, conf, input_dim, num_labels):
        '''
        Nnet constructor

        Args:
            conf: nnet configuration
            input_dim: network input dimension
            num_labels: number of target labels
        '''

        #get nnet structure configs
        self.conf = dict(conf.items('nnet'))

        #define location to save neural nets
        self.conf['savedir'] = (conf.get('directories', 'expdir')
                                + '/' + self.conf['name'])

        if not os.path.isdir(self.conf['savedir']):
            os.mkdir(self.conf['savedir'])
        if not os.path.isdir(self.conf['savedir'] + '/training'):
            os.mkdir(self.conf['savedir'] + '/training')

        #compute the input_dimension of the spliced features
        self.input_dim = input_dim * (2*int(self.conf['context_width']) + 1)

        if self.conf['batch_norm'] == 'True':
            activation = classifiers.activation.Batchnorm(None)
        else:
            activation = None

        #create the activation function
        if self.conf['nonlin'] == 'relu':
            activation = classifiers.activation.TfActivation(activation,
                                                             tf.nn.relu)

        elif self.conf['nonlin'] == 'sigmoid':
            activation = classifiers.activation.TfActivation(activation,
                                                             tf.nn.sigmoid)

        elif self.conf['nonlin'] == 'tanh':
            activation = classifiers.activation.TfActivation(activation,
                                                             tf.nn.tanh)

        elif self.conf['nonlin'] == 'linear':
            activation = classifiers.activation.TfActivation(activation,
                                                             lambda(x): x)

        else:
            raise Exception('unkown nonlinearity')

        if self.conf['l2_norm'] == 'True':
            activation = classifiers.activation.L2Norm(activation)

        if float(self.conf['dropout']) < 1:
            activation = classifiers.activation.Dropout(
                activation, float(self.conf['dropout']))

        #create a DNN
        self.dnn = DNN(
            num_labels, int(self.conf['num_hidden_layers']),
            int(self.conf['num_hidden_units']), activation,
            int(self.conf['add_layer_period']) > 0)

    def train(self, dispenser):
        '''
        Train the neural network

        Args:
            dispenser: a batchdispenser for training
        '''

        #get the validation set
        val_data, val_labels = zip(
            *[dispenser.get_batch()
              for _ in range(int(self.conf['valid_batches']))])

        val_data = list(itertools.chain.from_iterable(val_data))
        val_labels = list(itertools.chain.from_iterable(val_labels))

        dispenser.split()

        #compute the total number of steps
        num_steps = int(dispenser.num_batches *int(self.conf['num_epochs']))

        #set the step to the saving point that is closest to the starting step
        step = (int(self.conf['starting_step'])
                - int(self.conf['starting_step'])
                % int(self.conf['check_freq']))

        #go to the point in the database where the training was at checkpoint
        for _ in range(step):
            dispenser.skip_batch()

        if self.conf['numutterances_per_minibatch'] == '-1':
            numutterances_per_minibatch = dispenser.size
        else:
            numutterances_per_minibatch = int(
                self.conf['numutterances_per_minibatch'])

        #put the DNN in a training environment
        trainer = CrossEnthropyTrainer(
            self.dnn, self.input_dim, dispenser.max_input_length,
            dispenser.max_target_length,
            float(self.conf['initial_learning_rate']),
            float(self.conf['learning_rate_decay']),
            num_steps, numutterances_per_minibatch)

        #start the visualization if it is requested
        if self.conf['visualise'] == 'True':
            if os.path.isdir(self.conf['savedir'] + '/logdir'):
                shutil.rmtree(self.conf['savedir'] + '/logdir')

            trainer.start_visualization(self.conf['savedir'] + '/logdir')

        #start a tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        with tf.Session(graph=trainer.graph, config=config):
            #initialise the trainer
            trainer.initialize()

            #load the neural net if the starting step is not 0
            if step > 0:
                trainer.restore_trainer(self.conf['savedir']
                                        + '/training/step' + str(step))

            #do a validation step
            if val_data is not None:
                validation_loss = trainer.evaluate(val_data, val_labels)
                print 'validation loss at step %d: %f' % (step, validation_loss)
                validation_step = step
                trainer.save_trainer(self.conf['savedir']
                                     + '/training/validated')
                num_retries = 0

            #start the training iteration
            while step < num_steps:

                #get a batch of data
                batch_data, batch_labels = dispenser.get_batch()

                #update the model
                loss = trainer.update(batch_data, batch_labels)

                #print the progress
                print 'step %d/%d loss: %f' %(step, num_steps, loss)

                #increment the step
                step += 1

                #validate the model if required
                if (step%int(self.conf['valid_frequency']) == 0
                        and val_data is not None):

                    current_loss = trainer.evaluate(val_data, val_labels)
                    print 'validation loss at step %d: %f' %(step, current_loss)

                    if self.conf['valid_adapt'] == 'True':
                        #if the loss increased, half the learning rate and go
                        #back to the previous validation step
                        if current_loss > validation_loss:

                            #go back in the dispenser
                            for _ in range(step-validation_step):
                                dispenser.return_batch()

                            #load the validated model
                            trainer.restore_trainer(self.conf['savedir']
                                                    + '/training/validated')
                            trainer.halve_learning_rate()
                            step = validation_step

                            if num_retries == int(self.conf['valid_retries']):
                                print ('the validation loss is worse, '
                                       'terminating training')
                                break

                            print ('the validation loss is worse, returning to '
                                   'the previously validated model with halved '
                                   'learning rate')

                            num_retries += 1

                            continue

                        else:
                            validation_loss = current_loss
                            validation_step = step
                            num_retries = 0
                            trainer.save_trainer(self.conf['savedir']
                                                 + '/training/validated')

                #add a layer if its required
                if int(self.conf['add_layer_period']) > 0:
                    if (step%int(self.conf['add_layer_period']) == 0
                            and (step/int(self.conf['add_layer_period'])
                                 < int(self.conf['num_hidden_layers']))):

                        print 'adding layer, the model now holds %d/%d layers'%(
                            step/int(self.conf['add_layer_period']) + 1,
                            int(self.conf['num_hidden_layers']))

                        trainer.control_ops['add'].run()
                        trainer.control_ops['init'].run()

                        #do a validation step
                        validation_loss = trainer.evaluate(val_data, val_labels)
                        print 'validation loss at step %d: %f' % (
                            step, validation_loss)
                        validation_step = step
                        trainer.save_trainer(self.conf['savedir']
                                             + '/training/validated')
                        num_retries = 0

                #save the model if at checkpoint
                if step%int(self.conf['check_freq']) == 0:
                    trainer.save_trainer(self.conf['savedir'] + '/training/step'
                                         + str(step))


            #save the final model
            trainer.save_model(self.conf['savedir'] + '/final')

        #compute the state prior and write it to the savedir
        prior = dispenser.compute_target_count().astype(np.float32)
        prior = prior/prior.sum()

        np.save(self.conf['savedir'] + '/prior.npy', prior)

    def decode(self, reader, writer):
        '''
        compute pseudo likelihoods the testing set

        Args:
            reader: a feature reader object to read features to decode
            writer: a writer object to write likelihoods
        '''

        #create a decoder
        decoder = Decoder(self.dnn, self.input_dim, reader.max_input_length)

        #read the prior
        prior = np.load(self.conf['savedir'] + '/prior.npy')

        #start tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        with tf.Session(graph=decoder.graph, config=config):

            #load the model
            decoder.restore(self.conf['savedir'] + '/final')

            #feed the utterances one by one to the neural net
            while True:
                utt_id, utt_mat, looped = reader.get_utt()

                if looped:
                    break

                #compute predictions
                output = decoder(utt_mat)

                #get state likelihoods by dividing by the prior
                output = output/prior

                #floor the values to avoid problems with log
                np.where(output == 0, np.finfo(float).eps, output)

                #write the pseudo-likelihoods in kaldi feature format
                writer.write_next_utt(utt_id, np.log(output))

        #close the writer
        writer.close()
