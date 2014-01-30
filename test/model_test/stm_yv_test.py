'''
Created on 17 Jan 2014

@author: bryanfeeney
'''
from util.array_utils import normalizerows_ip, rowwise_softmax
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import model.ctm as ctm
import numpy as np
import numpy.random as rd
import scipy as sp
import scipy.linalg as la
import scipy.sparse as ssp
import scipy.sparse.linalg as sla
import unittest
import pickle as pkl

import model.stm_yv as stm

from model_test.sidetopic_test import matrix_normal
from math import ceil

DTYPE=np.float32

class Test(unittest.TestCase):
    
    def _sampleFromModel(self, D=200, T=100, K=10, F=12, P=8, avgWordsPerDoc = 500):
        '''
        Create a test dataset according to the model
        
        Params:
            T - Vocabulary size, the number of "terms". Must be a square number
            K - Observed topics
            P - Latent features
            F - Observed features
            D - Sample documents (each with associated features)
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
        vocab = np.zeros((K,T))
        for k in range(K):
            vocab[k,:] = rd.dirichlet(betaVec)
        
        # Geneate the shared covariance matrix
        sigT = rd.random((K,K))
        sigT = sigT.dot(sigT)
        sigT.flat[::K+1] += rd.random((K,)) * 4
        
        # Just link two topics
        sigT[K//2, K//3] = 3
        sigT[K//3, K//2] = 3
        
        sigT[4 * K//5, K//5] = 4
        sigT[K//5, 4 * K//5] = 4
        
        # Generate Y, then V, then A
        lfv = 0.1 # latent feature variance (for Y)
        fv  = 0.1 # feature variance (for A)
        
        Y = matrix_normal(np.zeros((K,P)),   lfv * np.eye(P), sigT)
        V = matrix_normal(np.zeros((P,F)),   fv * np.eye(F), lfv * np.eye(P))
        A = matrix_normal(Y.dot(V), fv * np.eye(F), sigT)
        
        # Generate the input features. Assume the features are multinomial and sparse
        # (not quite a perfect match for the twitter example: twitter is binary, this 
        # may not be)
        featuresDist  = [1. / F] * F 
        maxNonZeroFeatures = 3
        
        X = np.zeros((D,F), dtype=np.float32)
        for d in range(D):
            X[d,:] = rd.multinomial(maxNonZeroFeatures, featuresDist)
        X = ssp.csr_matrix(X)
        
        # Use the features and the matrix A to generate the topics and documents
        tpcs = rowwise_softmax (X.dot(A.T))
        
        docLens = rd.poisson(avgWordsPerDoc, (D,)).astype(np.float32)
        W = tpcs.dot(vocab)
        W *= docLens[:, np.newaxis]
        W = np.array(W, dtype=np.int32) # truncate word counts to integers
        W = ssp.csr_matrix(W)
        
        # Return the initialised model, the true parameter values, and the
        # generated observations
        return tpcs, vocab, docLens, X, W
        
    def testLikelihoodOnModelDerivedExample(self):
        print("Cross-validated likelihoods on model-derived example")
        
        rd.seed(0xBADB055) # Global init for repeatable test
        D, T, K, F, P = 200, 100, 10, 12, 8
        tpcs, vocab, docLens, X, W = self._sampleFromModel()
        
        W = W.astype(DTYPE)
        X = X.astype(DTYPE)
        
        # Create the cross-validation folds
        folds     = 5
        foldSize  = ceil(D / 5)
        querySize = foldSize
        trainSize = D - querySize
        
        for fold in range(1):
            start = fold * foldSize
            end   = start + trainSize
            
            trainSet = np.arange(start,end) % D
            querySet = np.arange(end, end + querySize) % D
            
            X_train, W_train = X[trainSet,:], W[trainSet,:]
            X_query, W_query = X[querySet,:], W[querySet,:]
            
            model = stm.newModelAtRandom(X_train, W_train, P, K, 0.1, 0.1, dtype=np.float32)
            queryState = stm.newQueryState(W_train, model)
            
            plan  = stm.newTrainPlan(iterations=40, plot=True, logFrequency=1)
            model, queryState = stm.train(W_train, X_train, model, queryState, plan)
            
            self._plotCov(model)
            
            trainSetLikely = stm.log_likelihood(W_train, model, queryState)
            
            
            print("Fold %d: Train-set Likelihood: %12f \t Query-set Likelihood: TODO" % (fold, trainSetLikely))
            print("")
            
        print("End of Test")
        
    def _plotCov(self, model):
        fig, ax = plt.subplots(figsize=(11,9))

        im = plt.imshow(model.sigT, interpolation="none")

        cb = fig.colorbar(im, ax=ax)
        ax.set_title("Covariance Matrix")
        plt.show()
    
    
    def _doTest (self, W, model, queryState, trainPlan):
        D,_ = W.shape
        recons = queryState.means.dot(model.vocab)
        reconsErr = 1./D * np.sum((np.asarray(W.todense()) - recons) * (np.asarray(W.todense()) - recons))
        
        print ("Initial bound is %f\n\n" % ctm.var_bound(W, model, queryState))
        print ("Initial reconstruction error is %f\n\n" % reconsErr)
        
        model, query = ctm.train (W, model, queryState, trainPlan)
        ones = np.ones((3,3))
        for k in range(model.K):
            plt.subplot(2, 3, k)
            plt.imshow(ones - model.vocab[k,:].reshape((3,3)), interpolation="none", cmap = cm.Greys_r)
        plt.show()
        
        recons = queryState.means.dot(model.vocab)
        reconsErr = 1./D * np.sum((np.asarray(W.todense()) - recons) * (np.asarray(W.todense()) - recons))
        print ("Final reconstruction error is %f\n\n" % reconsErr)
        

    def _testOnRealData(self):
        path = "/Users/bryanfeeney/Desktop/SmallerDB-NoCJK-WithFeats"
        with open(path + "/words-by-author.pkl", 'rb') as f:
            user_dict, d, W = pkl.load(f)
        
        if W.dtype != DTYPE:
            W = W.astype(DTYPE)
        
        K = 30
        model      = ctm.newModelAtRandom(W, K, dtype=DTYPE)
        queryState = ctm.newQueryState(W, model)
        trainPlan  = ctm.newTrainPlan(iterations=200, plot=True, logFrequency=1)
        
        model, query = ctm.train (W, model, queryState, trainPlan)
        with open("/Users/bryanfeeney/Desktop/test_result.pkl", "wb") as f:
            pkl.dump ((model, query), f)
    
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