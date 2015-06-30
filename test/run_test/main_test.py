'''
Created on 27 Nov 2013

@author: bryanfeeney
'''
import unittest
import pickle as pkl
import tempfile as tmp
import cProfile

from model_test.stm_yv_test import sampleFromModel
from run.main import run, ModelNames, \
    Rtm, LdaGibbs, LdaVb, Mtm, StmYvBohning, StmYvBouchard, CtmBohning, CtmBouchard
from model.evals import Perplexity, MeanAveragePrecAllDocs, MeanPrecRecAtMAllDocs

AclPath = "/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100/"
AclWordPath  = AclPath + "words-freq.pkl"
AclFeatsPath = AclPath + "feats.pkl"
AclCitePath  = AclPath + "ref.pkl"

TweetsPath = "/Users/bryanfeeney/iCloud/Datasets/Tweets/Cluster2015-06-24/AuthorTime750/"
TweetsWordPath = TweetsPath + "words-cleaned.pkl"
TweetsFeatPath = TweetsPath + "side-cleaned.pkl"

AuthorTweetsWordPath = TweetsPath + "words-by-author.pkl"

NipsPath = "/Users/bryanfeeney/iCloud/Datasets/NIPS-from-pryor-Sep15/"
NipsWordPath = NipsPath + "W_ar.pkl"
NipsFeatPath = NipsPath + "X_ar.pkl"

Acl, Tweets, AuthorTweets, Nips = 0, 1, 2, 3
WordsPath = [AclWordPath,  TweetsWordPath, AuthorTweetsWordPath, NipsWordPath]
FeatsPath = [AclFeatsPath, TweetsFeatPath, None,                 NipsFeatPath]
CitesPath = [AclCitePath,  None,           None,                 None]

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
        
        print ("New Version")

        DataSetName = AuthorTweets
        Folds = 5
        K,P = 40, 50
        TrainIters, QueryIters, LogFreq = 1000, 100, 5,
        Debug = True

        modelFileses = []
        for modelName in [ LdaVb ]: #ModelNames:
            cmdline = '' \
                    + (' --debug '         + str(Debug) if Debug else "") \
                    + ' --model '          + modelName \
                    + ' --dtype '          + 'f8:f8'      \
                    + ' --num-topics '     + str(K)    \
                    + ' --num-lat-feats '  + str(P) \
                    + ' --log-freq '       + str(LogFreq)       \
                    + ' --eval '           + 'perplexity'  \
                    + ' --iters '          + str(TrainIters)      \
                    + ' --query-iters '    + str(QueryIters)      \
                    + ' --folds '          + str(Folds)      \
                    + ' --words '          + WordsPath[DataSetName] \
                    + (' --feats '         + FeatsPath[DataSetName] if FeatsPath[DataSetName] is not None else "") \
                    + (' --links '         + CitesPath[DataSetName] if CitesPath[DataSetName] is not None else "") \
                    + ' --out-model '      + '/Users/bryanfeeney/Desktop/acl-out'
#                     + ' --words '          + '/Users/bryanfeeney/Dropbox/Datasets/ACL/words.pkl' \
#                     + ' --words '          + '/Users/bryanfeeney/Desktop/NIPS-from-pryor-Sep15/W_ar.pkl'
#                      + ' --words '          + '/Users/bryanfeeney/Desktop/Dataset-Sep-2014/words.pkl' \
#                      + ' --feats '          + '/Users/bryanfeeney/Desktop/Dataset-Sep-2014/side.pkl'
#                    + ' --words '          + wordsFile \
#                    + ' --feats '          + featsFile 
#                    + ' --words '          + '/Users/bryanfeeney/Desktop/Tweets600/words-by-author.pkl' \
      
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
    import sys;sys.argv = ['', 'Test.testUyv']
    unittest.main()

