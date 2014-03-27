'''
Created on 27 Nov 2013

@author: bryanfeeney
'''
import unittest
import pickle as pkl
import tempfile as tmp

from model_test.stm_yv_test import sampleFromModel
from run.main import run

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


    def testAlgorithms(self):
        D, T, K, Q, F, P, avgWordsPerDoc = 200, 100, 10, 6, 12, 8, 500
        tpcs, vocab, docLens, X, W = sampleFromModel(D, T, K, F, P, avgWordsPerDoc)
        
        wordsFile, featsFile, modelFileDir = tmpFiles()
        with open(wordsFile, 'wb') as f:
            pkl.dump(W, f)
        with open(featsFile, 'wb') as f:
            pkl.dump(X, f)
        
#        K, P = 10, 75
        modelFileses = []
        for algorithm in ["stm_yv", "ctm"]:
            for bound in ["bouchard", "bohning"]:
                cmdline = '' \
                        + ' --debug True' \
                        + ' --model '          + algorithm + '_' + bound \
                        + ' --num-topics '     + str(K)    \
                        + ' --num-lat-topics ' + str(Q)    \
                        + ' --num-lat-feats '  + str(P)    \
                        + ' --eval '           + 'likely'  \
                        + ' --out-model '      + modelFileDir \
                        + ' --log-freq '       + '10'     \
                        + ' --iters '          + '50'     \
                        + ' --query-iters '    + '10'      \
                        + ' --min-vb-change '  + '0.00001' \
                        + ' --topic-var '      + '0.01'    \
                        + ' --feat-var '       + '0.01'    \
                        + ' --lat-topic-var '  + '0.1'       \
                        + ' --lat-feat-var '   + '0.1'       \
                        + ' --folds '          + '5'       \
                        + ' --words '          + wordsFile \
                        + ' --feats '          + featsFile 
#                        + ' --words '          + '/Users/bryanfeeney/Desktop/NIPS/W_ar.pkl' \
#                        + ' --feats '          + '/Users/bryanfeeney/Desktop/NIPS/X_ar.pkl'
      
                modelFileses.extend (run(cmdline.strip().split(' ')))
        
        modelFileses.insert(0, wordsFile)
        modelFileses.insert(1, featsFile)
        print ("Files can be found in:" + "\n\t".join(modelFileses))
        
    
    def _testLoadResult(self):
        path = "/Users/bryanfeeney/Desktop/out.sample/ctm_bouchard_k_50_20140223_1719.pkl"
        with open (path, 'rb') as f:
            (order, boundItrses, boundValses, models, trainTopicses, queryTopicses) = pkl.load(f)
        print (str(boundItrses[0]))
        print (models[0].name)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testUyv']
    unittest.main()
    