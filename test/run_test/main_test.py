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

AclPath = "/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100.clean/"
_AclWordPath  = AclPath + "words-freq.pkl"
_AclFeatsPath = AclPath + "feats.pkl"
_AclCitePath  = AclPath + "ref.pkl"

NipsPath = "/Users/bryanfeeney/iCloud/Datasets/NIPS-from-pryor-Sep15/"
_NipsWordPath = NipsPath + "W_ar.pkl"
_NipsFeatPath = NipsPath + "X_ar.pkl"


Tweets2500Path = "/Users/bryanfeeney/iCloud/Datasets/Tweets/Tweets-2.5m/"
_Tweets2500WordPath = Tweets2500Path + "words.pkl"
_Tweets2500FeatPath = Tweets2500Path + "side.pkl"

_AuthorTweets2500WordPath     = Tweets2500Path + "words-by-author.pkl"
_AuthorTweets2500FreqWordPath = Tweets2500Path + "words-by-author-freq.pkl"
_Tweets2500FreqWordPath       = Tweets2500Path + "words-freq.pkl"


Tweets750Path = "/Users/bryanfeeney/iCloud/Datasets/Tweets/Cluster2015-06-24/AuthorTime750/"
_Tweets750WordPath = Tweets750Path + "words-cleaned.pkl"
_Tweets750FeatPath = Tweets750Path + "side-cleaned.pkl"

_AuthorTweets750WordPath     = Tweets750Path + "words-by-author.pkl"
_AuthorTweets750FreqWordPath = Tweets750Path + "words-by-author-freq.pkl"
_Tweets750FreqWordPath       = Tweets750Path + "words-cleaned-freq.pkl"

Tweets800Path = "/Users/bryanfeeney/Desktop/SmallerDB-NoCJK-WithFeats-Fixed/"
_Tweets800WordPath = Tweets800Path + "words.pkl"
_Tweets800FeatPath = Tweets800Path + "side.pkl"

_AuthorTweets800WordPath     = Tweets800Path + "words-by-author.pkl"
_AuthorTweets800FreqWordPath = Tweets800Path + "words-by-author-freq.pkl"
_Tweets800FreqWordPath       = Tweets800Path + "words-freq.pkl"

Tweets500Path = "/Users/bryanfeeney/iCloud/Datasets/Tweets/AuthorTime/"
_Tweets500WordPath = Tweets500Path + "words.pkl"
_Tweets500FeatPath = Tweets500Path + "side.pkl"

_Author500TweetsWordPath     = Tweets500Path + "words-by-author.pkl"
_Author500TweetsFreqWordPath = Tweets500Path + "words-by-author-freq.pkl"
_Tweets500FreqWordPath       = Tweets500Path + "words-cleaned-freq.pkl"

 # Pick either 500 or 750 or 800 or 2500
_TweetsWordPath = _Tweets2500WordPath
_TweetsFeatPath = _Tweets2500FeatPath

_AuthorTweetsWordPath     = _AuthorTweets2500WordPath
_AuthorTweetsFreqWordPath = _AuthorTweets2500FreqWordPath
_TweetsFreqWordPath       = _Tweets2500FreqWordPath


Acl, AclNoLinks, TweetsAll, TweetsFreq, AuthorTweetsAll, AuthorTweetsFreq, Nips = 0, 1, 2, 3, 4, 5, 6
WordsPath = [_AclWordPath,  _AclWordPath,  _TweetsWordPath, _TweetsFreqWordPath, _AuthorTweetsWordPath, _AuthorTweetsFreqWordPath, _NipsWordPath]
FeatsPath = [_AclFeatsPath, _AclFeatsPath, _TweetsFeatPath, _TweetsFeatPath,     None,                 None,                     _NipsFeatPath]
CitesPath = [_AclCitePath,  None,         None,           None,               None,                 None,                     None]

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

        Folds, ExecutedFoldCount = 5, 1
        K,P = 25, 50
        TrainIters, QueryIters, LogFreq = 2000, 500, 10
        PriorCov = 0.001
        VocabPrior = 0.01
        Debug = False

        modelFileses = []
        for DataSetName in [TweetsFreq]:
            for k in [100]: # [10, 25, 100]:
                for modelName in [ StmYvBohning ]: #ModelNames:
                    cmdline = '' \
                            + (' --debug '         + str(Debug) if Debug else "") \
                            + ' --model '          + modelName \
                            + ' --dtype '          + 'f8:f8'      \
                            + ' --num-topics '     + str(k)    \
                            + ' --num-lat-feats '  + str(P) \
                            + ' --log-freq '       + str(LogFreq)       \
                            + ' --eval '           + 'perplexity'  \
                            + ' --iters '          + str(TrainIters)      \
                            + ' --query-iters '    + str(QueryIters)      \
                            + ' --folds '          + str(Folds)      \
                            + ' --truncate-folds ' + str(ExecutedFoldCount)      \
                            + ' --words '          + WordsPath[DataSetName] \
                            + (' --feats '         + FeatsPath[DataSetName] if FeatsPath[DataSetName] is not None else "") \
                            + (' --links '         + CitesPath[DataSetName] if CitesPath[DataSetName] is not None else "") \
                            + ' --topic-var '      + str(PriorCov) \
                            + ' --feat-var '       + str(PriorCov) \
                            + ' --lat-topic-var '  + str(PriorCov) \
                            + ' --lat-feat-var '   + str(PriorCov) \
                            + ' --vocab-prior '    + str(VocabPrior) \
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

