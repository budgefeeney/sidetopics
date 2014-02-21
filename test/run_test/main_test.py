'''
Created on 27 Nov 2013

@author: bryanfeeney
'''
import unittest
import pickle as pkl
import tempfile as tmp

from run.main import run

from model_test.stm_yv_test import sampleFromModel

def tmpFiles():
    '''
    Returns files in the temporary directory for storing the DxT matrix of
    word counts, the DxF matrix of features, the file which stores the model
    details, and the file containing a plot of the variational bounds.
    '''
    tmpDir = tmp.gettempdir()
    return tmpDir + '/words.pkl', tmpDir + '/feats.pkl', tmpDir

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testUyv(self):
        D, T, K, Q, F, P, avgWordsPerDoc = 200, 100, 10, 6, 12, 8, 500
        tpcs, vocab, docLens, X, W = sampleFromModel(D, T, K, F, P, avgWordsPerDoc)
        
        wordsFile, featsFile, modelFile = tmpFiles()
        with open(wordsFile, 'wb') as f:
            pkl.dump(W, f)
        with open(featsFile, 'wb') as f:
            pkl.dump(X, f)
        
        
        K, Q, P = 40, 10, 50
        cmdline = '' \
                + ' --model '          + 'ctm_bouchard'      \
                + ' --num-topics '     + str(K)    \
                + ' --num-lat-topics ' + str(Q)    \
                + ' --num-lat-feats '  + str(P)    \
                + ' --eval '           + 'likely'  \
                + ' --out-model '      + modelFile \
                + ' --out-plot '       + plotFile  \
                + ' --log-freq '       + '100'     \
                + ' --iters '          + '500'     \
                + ' --query-iters '    + '50'      \
                + ' --min-vb-change '  + '0.00001' \
                + ' --topic-var '      + '0.01'    \
                + ' --feat-var '       + '0.01'    \
                + ' --lat-topic-var '  + '0.1'       \
                + ' --lat-feat-var '   + '0.1'       \
                + ' --folds '          + '5'       \
                + ' --feats '          + featsFile \
                + ' --words '          + wordsFile
#                 + ' --feats '          + '/Users/bryanfeeney/Desktop/SmallDB2/side.pkl' \
#                 + ' --words '          + '/Users/bryanfeeney/Desktop/SmallDB2/words.pkl'
        
        run(cmdline.strip().split(' '))
        print ("Files can be found in %s, %s, %s, %s" % ( wordsFile, featsFile, modelFile, plotFile))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testUyv']
    unittest.main()
    