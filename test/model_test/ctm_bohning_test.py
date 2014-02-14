'''
Created on 17 Jan 2014

@author: bryanfeeney
'''
from util.array_utils import normalizerows_ip
from util.sigmoid_utils import rowwise_softmax
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import model.ctm_bohning as ctm
import numpy as np
import numpy.random as rd
import scipy as sp
import scipy.linalg as la
import scipy.sparse as ssp
import scipy.sparse.linalg as sla
import unittest
import pickle as pkl



DTYPE=np.float32

class Test(unittest.TestCase):

    def _testOnModelHandcraftedData(self):
        #
        # Create the vocab
        #
        T = 3 * 3
        K = 5
        
        # Horizontal bars
        vocab1 = ssp.coo_matrix(([1, 1, 1], ([0, 0, 0], [0, 1, 2])), shape=(3,3)).todense()
        #vocab2 = ssp.coo_matrix(([1, 1, 1], ([1, 1, 1], [0, 1, 2])), shape=(3,3)).todense()
        vocab3 = ssp.coo_matrix(([1, 1, 1], ([2, 2, 2], [0, 1, 2])), shape=(3,3)).todense()
        
        # Vertical bars
        vocab4 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [0, 0, 0])), shape=(3,3)).todense()
        #vocab5 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [1, 1, 1])), shape=(3,3)).todense()
        vocab6 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [2, 2, 2])), shape=(3,3)).todense()
        
        # Diagonals
        vocab7 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3,3)).todense()
        #vocab8 = ssp.coo_matrix(([1, 1, 1], ([2, 1, 0], [0, 1, 2])), shape=(3,3)).todense()
        
        # Put together
        T = vocab1.shape[0] * vocab1.shape[1]
        vocabs = [vocab1, vocab3, vocab4, vocab6, vocab7]
        
        # Create a single matrix with the flattened vocabularies
        vocabVectors = []
        for vocab in vocabs:
            vocabVectors.append (np.squeeze(np.asarray (vocab.reshape((1,T)))))
        
        vocab = normalizerows_ip(np.array(vocabVectors, dtype=DTYPE))
        
        # Plot the vocab
        ones = np.ones(vocabs[0].shape)
        for k in range(K):
            plt.subplot(2, 3, k)
            plt.imshow(ones - vocabs[k], interpolation="none", cmap = cm.Greys_r)
        plt.show()
        
        #
        # Create the corpus
        #
        rd.seed(0xC0FFEE)
        D = 1000

        # Make sense (of a sort) of this by assuming that these correspond to
        # Kittens    Omelettes    Puppies    Oranges    Tomatoes    Dutch People    Basketball    Football
        #topicMean = np.array([10, 25, 5, 15, 5, 5, 10, 25])
#        topicCovar = np.array(\
#            [[ 100,    5,     55,      20,     5,     15,      4,      0], \
#             [ 5,    100,      5,      10,    70,      5,      0,      0], \
#             [ 55,     5,    100,       5,     5,     10,      0,      5], \
#             [ 20,    10,      5,     100,    30,     30,     20,     10], \
#             [ 5,     70,      5,     30,    100,      0,      0,      0], \
#             [ 15,     5,     10,     30,      0,    100,     10,     40], \
#             [ 4,      0,      0,     20,      0,     10,    100,     20], \
#             [ 0,      0,      5,     10,      0,     40,     20,    100]], dtype=DTYPE) / 100.0

        topicMean = np.array([25, 15, 40, 5, 15])
        self.assertEqual(100, topicMean.sum())
        topicCovar = np.array(\
            [[ 100,    5,     55,      20,     5     ], \
             [ 5,    100,      5,      10,    70     ], \
             [ 55,     5,    100,       5,     5     ], \
             [ 20,    10,      5,     100,    30     ], \
             [ 5,     70,      5,     30,    100     ], \
             ], dtype=DTYPE) / 100.0
 
        
        meanWordCount = 80
        wordCounts = rd.poisson(meanWordCount, size=D)
        topicDists = rd.multivariate_normal(topicMean, topicCovar, size=D)
        W = topicDists.dot(vocab) * wordCounts[:, np.newaxis]
        W = ssp.csr_matrix (W.astype(DTYPE))
        
        #
        # Train the model
        #
        model      = ctm.newModelAtRandom(W, K, dtype=DTYPE)
        queryState = ctm.newQueryState(W, model)
        trainPlan  = ctm.newTrainPlan(iterations=65, plot=True, logFrequency=1)
        
        self.assertTrue (0.99 < np.sum(model.topicMean) < 1.01)
        
        return self._doTest (W, model, queryState, trainPlan)

    def _sampleFromModel(self, D=200, T=100, K=10, avgWordsPerDoc = 500):
        '''
        Create a test dataset according to the model
        
        Params:
            D - Sample documents (each with associated features)
            T - Vocabulary size, the number of "terms". Must be a square number
            K - Observed topics
            avgWordsPerDoc - average number of words per document generated (Poisson)
        
        Returns:
            modelState - a model state object configured for training
            tpcs       - the matrix of per-document topic distribution
            vocab      - the matrix of per-topic word distributions
            docLens    - the vector of document lengths
            X          - the DxF side information matrix
            W          - The DxW word matrix
        '''
        
        # Generate vocab
        beta = 0.1
        betaVec = np.ndarray((T,))
        betaVec.fill(beta)
        vocab = rd.dirichlet(betaVec, size=K)
        
        # Geneate the shared covariance matrix
        # ...no real structure in this.
        sigT = rd.random((K,K))
        sigT = sigT.dot(sigT)
        
        # Generate topic mean
        alpha = 1
        alphaVec = np.ndarray((K,))
        alphaVec.fill(alpha)
        topicMean = rd.dirichlet(alphaVec)
        
        # Generate the actual topics.
        tpcs = rd.multivariate_normal(topicMean, sigT, size=D)
        tpcs = rowwise_softmax(tpcs)
        
        # Generate the corpus
        docLens = rd.poisson(avgWordsPerDoc, (D,)).astype(np.float32)
        W = tpcs.dot(vocab)
        W *= docLens[:, np.newaxis]
        W = np.array(W, dtype=np.int32) # truncate word counts to integers
        W = ssp.csr_matrix(W)
        
        # Return the initialised model, the true parameter values, and the
        # generated observations
        return tpcs, vocab, docLens, W
        
    def _testOnModelDerivedExample(self):
        print("Cross-validated likelihoods on model-derived example")
        
        rd.seed(0xBADB055) # Global init for repeatable test
        D, T, K = 1000, 100, 7 # Document count, vocabularly size ("term count") and topic count
        tpcs, vocab, docLens, W = self._sampleFromModel()
        
        W = W.astype(DTYPE)
        
        plt.imshow(vocab, interpolation="none", cmap = cm.Greys_r)
        plt.show()
            
        #
        # Train the model
        #
        model      = ctm.newModelAtRandom(W, K, dtype=DTYPE)
        queryState = ctm.newQueryState(W, model)
        trainPlan  = ctm.newTrainPlan(iterations=150, plot=True, logFrequency=1)
        
        return self._doTest (W, model, queryState, trainPlan)
        print("End of Test")
        
    def _plotCov(self, model):
        fig, ax = plt.subplots(figsize=(11,9))

        im = plt.imshow(model.sigT, interpolation="none")

        cb = fig.colorbar(im, ax=ax)
        ax.set_title("Covariance Matrix")
        plt.show()
        
    
    
    def _doTest (self, W, model, queryState, trainPlan):
        D,_ = W.shape
        recons = rowwise_softmax(queryState.means).dot(model.vocab)
        reconsErr = 1./D * np.sum((np.asarray(W.todense()) - recons) * (np.asarray(W.todense()) - recons))
        
        print ("Initial bound is %f\n\n" % ctm.var_bound(W, model, queryState))
        print ("Initial reconstruction error is %f\n\n" % reconsErr)
        
        model, query = ctm.train (W, None, model, queryState, trainPlan)
        plt.imshow(model.vocab, interpolation="none", cmap = cm.Greys_r)
        plt.show()
        
        recons = rowwise_softmax(queryState.means).dot(model.vocab)
        reconsErr = 1./D * np.sum((np.asarray(W.todense()) - recons) * (np.asarray(W.todense()) - recons))
        print ("Final reconstruction error is %f\n\n" % reconsErr)
    

    def testOnRealData(self):
        rd.seed(0xC0FFEE)
        dtype = np.float64
        
        path = "/Users/bryanfeeney/Desktop/SmallerDB-NoCJK-WithFeats-Fixed"
        with open(path + "/words-by-author.pkl", 'rb') as f:
            user_dict, d, W = pkl.load(f)
        
        if W.dtype != dtype:
            W = W.astype(dtype)
        D,T = W.shape
       
        # Initialise the model  
        K = 20
        model      = ctm.newModelAtRandom(W, K, dtype=dtype)
        queryState = ctm.newQueryState(W, model)
        trainPlan  = ctm.newTrainPlan(iterations=200, plot=True, logFrequency=1)
        
        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query = ctm.train (W, None, model, queryState, trainPlan)
        with open("/Users/bryanfeeney/Desktop/author_ctm_result-2v.pkl", "wb") as f:
            pkl.dump ((model, query), f)
    
        # Print out the most likely topic words
        topWordCount = 100
        kTopWordInds = []
        for k in range(K):
            topWordInds = self.topWordInds(d, model.vocab[k,:], topWordCount)
            kTopWordInds.append(topWordInds)
        
        print ("Perplexity: %f\n\n" % ctm.perplexity(W, model, query))
        print ("\t\t".join (["Topic " + str(k) for k in range(K)]))
        print ("\n".join ("\t".join (d[kTopWordInds[k][c]] + "\t%0.4f" % model.vocab[k][kTopWordInds[k][c]] for k in range(K)) for c in range(topWordCount)))
        
    def topWords (self, wordDict, vocab, count=10):
        return [wordDict[w] for w in self.topWordInds(wordDict, vocab, count)]

    
    def topWordInds (self, wordDict, vocab, count=10):
        return vocab.argsort()[-count:][::-1]
    
    def printTopics(self, wordDict, vocab, count=10):
        words = vocab.argsort()[-count:][::-1]
        for wordIdx in words:
            print("%s" % wordDict[wordIdx])
        print("")


def truncate(word, max_len=12):      
    return word if len(word) < max_len else word[:(max_len-3)] + '...'
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()