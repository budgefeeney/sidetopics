'''
Created on 27 Nov 2013

@author: bryanfeeney
'''
import unittest
import pickle as pkl
import tempfile as tmp

from run.main import run

from model_test.sidetopic_uyv_test import sampleFromModel as sample_uyv_dataset

def tmpFiles():
    '''
    Returns files in the temporary directory for storing the DxT matrix of
    word counts, the DxF matrix of features, the file which stores the model
    details, and the file containing a plot of the variational bounds.
    '''
    tmpDir = tmp.gettempdir()
    return tmpDir + '/words.pkl', tmpDir + '/feats.pkl', tmpDir + '/model.pkl', tmpDir + '/plot'

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testUyv(self):
        D, T, K, Q, F, P, avgWordsPerDoc = 200, 100, 10, 6, 12, 8, 500
        modelState, _, _, _, X, W = sample_uyv_dataset(D, T, K, Q, F, P, avgWordsPerDoc)
        
        wordsFile, featsFile, modelFile, plotFile = tmpFiles()
        with open(wordsFile, 'wb') as f:
            pkl.dump(W, f)
        with open(featsFile, 'wb') as f:
            pkl.dump(X, f)
        
        cmdline = '' \
                + ' --model '          + 'uy'  \
                + ' --num-topics '     + str(K) \
                + ' --num-lat-topics ' + str(Q) \
                + ' --num-lat-feats '  + str(P) \
                + ' --feats '          + '/Users/bryanfeeney/Dropbox/SideTopicDatasets/side-short.pkl' \
                + ' --words '          + '/Users/bryanfeeney/Dropbox/SideTopicDatasets/words-short.pkl' \
                + ' --eval '           + 'likely'  \
                + ' --out-model '      + modelFile \
                + ' --out-plot '       + plotFile  \
                + ' --log-freq '       + '30'   \
                + ' --iters '          + '200'  \
                + ' --query-iters '    + '50'   \
                + ' --min-vb-change '  + '0.001'    \
                + ' --topic-var '      + '0.01' \
                + ' --feat-var '       + '0.01' \
                + ' --lat-topic-var '  + '1'    \
                + ' --lat-feat-var '   + '1'    \
                + ' --folds '          + '5'
        
        
        
        run(cmdline.strip().split(' '))
        print ("Files can be found in %s, %s, %s, %s" % ( wordsFile, featsFile, modelFile, plotFile))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testUyv']
    unittest.main()
    