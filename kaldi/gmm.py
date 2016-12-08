'''@file gmm.py
contains the functionality for Kaldi GMM training, aligning and testing'''

from abc import ABCMeta, abstractproperty
import os

class GMM(object):
    '''an abstract class for a kaldi GMM'''
    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''
        KaldiGMM constructor

        Args:
            conf: the general configurations
        '''

        self.conf = conf

    def train(self):
        '''train the GMM'''

        #save the current dir
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(self.conf.get('directories', 'kaldi_egs'))

        #train the GMM
        os.system('%s --cmd %s --config %s/config/%s %s %s %s %s %s' %(
            self.trainscript, self.conf.get('general', 'cmd'), current_dir,
            self.conf_file, self.trainops,
            self.conf.get('directories', 'train_features')
            + '/' + self.conf.get('gmm-features', 'name'),
            self.conf.get('directories', 'language'),
            self.parent_gmm_alignments,
            self.conf.get('directories', 'expdir') + '/' + self.name))

        #build the decoding graphs
        os.system('utils/mkgraph.sh %s %s %s %s/graph' % (
            self.graphopts, self.conf.get('directories', 'language_test'),
            self.conf.get('directories', 'expdir') + '/' + self.name,
            self.conf.get('directories', 'expdir') + '/' + self.name))

        #go back to working dir
        os.chdir(current_dir)

    def align(self):
        '''use the GMM to align the training utterances'''

        #save the current dir
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(self.conf.get('directories', 'kaldi_egs'))

        #do the alignment
        os.system('''steps/align_si.sh --nj %s --cmd %s
            --config %s/config/ali_%s %s %s %s %s/ali''' % (
                self.conf.get('general', 'num_jobs'),
                self.conf.get('general', 'cmd'), current_dir, self.conf_file,
                self.conf.get('directories', 'train_features') + '/'
                + self.conf.get('gmm-features', 'name'),
                self.conf.get('directories', 'language'),
                self.conf.get('directories', 'expdir') + '/' + self.name,
                self.conf.get('directories', 'expdir') + '/' + self.name))

        #convert alignments (transition-ids) to pdf-ids
        for i in range(int(self.conf.get('general', 'num_jobs'))):
            os.system('''gunzip -c %s/ali/ali.%d.gz | ali-to-pdf
                %s/ali/final.mdl ark:- ark,t:- | gzip >  %s/ali/pdf.%d.gz''' % (
                    self.conf.get('directories', 'expdir') + '/' + self.name,
                    i+1, self.conf.get('directories', 'expdir') + '/'
                    + self.name, self.conf.get('directories', 'expdir')
                    + '/' + self.name, i+1))

        #go back to working dir
        os.chdir(current_dir)

    def test(self):
        '''test the GMM on the testing set'''

        #save the current dir
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(self.conf.get('directories', 'kaldi_egs'))

        os.system('''steps/decode.sh --cmd %s --nj %s %s/graph %s %s/decode
        | tee %s/decode.log || exit 1;''' % (
            self.conf.get('general', 'cmd'),
            self.conf.get('general', 'num_jobs'),
            self.conf.get('directories', 'expdir') + '/' + self.name,
            self.conf.get('directories', 'test_features') + '/'
            + self.conf.get('gmm-features', 'name'),
            self.conf.get('directories', 'expdir') + '/' + self.name,
            self.conf.get('directories', 'expdir') + '/'  + self.name))

        #go back to working dir
        os.chdir(current_dir)

    @abstractproperty
    def name(self):
        '''the name of the GMM'''
        pass

    @abstractproperty
    def trainscript(self):
        '''the script used for training the GMM'''
        pass

    @abstractproperty
    def conf_file(self):
        '''the configuration file for this GMM'''
        pass

    @abstractproperty
    def parent_gmm_alignments(self):
        '''the path to the parent GMM model (empty for monophone GMM)'''
        pass

    @abstractproperty
    def trainops(self):
        '''the extra options for GMM training'''
        pass

    @abstractproperty
    def graphopts(self):
        '''the extra options for the decoding graph creation'''
        pass

class MonoGmm(GMM):
    ''' a class for the monophone GMM'''

    @property
    def name(self):
        return self.conf.get('mono_gmm', 'name')

    @property
    def trainscript(self):
        return 'steps/train_mono.sh'

    @property
    def conf_file(self):
        return 'mono.conf'

    @property
    def parent_gmm_alignments(self):
        return ''

    @property
    def trainops(self):
        return '--nj %s' % self.conf.get('general', 'num_jobs')

    @property
    def graphopts(self):
        return '--mono'

class TriGmm(GMM):
    '''a class for the triphone GMM'''

    @property
    def name(self):
        return self.conf.get('tri_gmm', 'name')

    @property
    def trainscript(self):
        return 'steps/train_deltas.sh'

    @property
    def conf_file(self):
        return 'tri.conf'

    @property
    def parent_gmm_alignments(self):
        return (self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('mono_gmm', 'name') + '/ali')

    @property
    def trainops(self):
        return (self.conf.get('tri_gmm', 'num_leaves') + ' '
                + self.conf.get('tri_gmm', 'tot_gauss'))

    @property
    def graphopts(self):
        return ''

class LdaGmm(GMM):
    '''a class for the LDA+MLLT GMM'''

    @property
    def name(self):
        return self.conf.get('lda_mllt', 'name')

    @property
    def trainscript(self):
        return 'steps/train_lda_mllt.sh'

    @property
    def conf_file(self):
        return 'lda_mllt.conf'

    @property
    def parent_gmm_alignments(self):
        return (self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('tri_gmm', 'name') + '/ali')

    @property
    def trainops(self):
        return '--context-opts "--context_width=%s"'% (
            self.conf.get('lda_mllt', 'context_width') + ' '
            +  self.conf.get('lda_mllt', 'num_leaves') + ' '
            + self.conf.get('lda_mllt', 'tot_gauss'))

    @property
    def graphopts(self):
        return ''
