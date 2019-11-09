'''
Created on 17 Jan 2014

@author: bryanfeeney
'''
from sidetopics.util.array_utils import normalizerows_ip
from sidetopics.util.sigmoid_utils import rowwise_softmax
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sidetopics.model.ctm as ctm
import numpy as np
import numpy.random as rd
import scipy as sp
import scipy.linalg as la
import scipy.sparse as ssp
import scipy.sparse.linalg as sla
import unittest
import pickle as pkl
import time

from math import ceil 

from run.main import newModelFileFromModel


DTYPE=np.float64

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
        trainPlan  = ctm.newTrainPlan(iterations=65, logFrequency=1)
        
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
        useDiagonalPriorCov = True
        
        rd.seed(0xBADB055) # Global init for repeatable test
        D, T, K = 1000, 100, 7 # Document count, vocabularly size ("term count") and topic count
        tpcs, vocab, docLens, W = self._sampleFromModel(D, T, K)
        
        W = W.astype(DTYPE)
        
        plt.imshow(vocab, interpolation="none", cmap = cm.Greys_r)
        plt.show()
        
        
        # Create the cross-validation folds
        folds     = 5
        foldSize  = ceil(D / 5)
        querySize = foldSize
        trainSize = D - querySize
        
        for useDiagonalPriorCov in [False, True]:
            trainLikely = []
            trainWordCount = []
            queryLikely = []
            queryWordCount = []
            
            for fold in range(folds):
                # Split the datasets
                start = fold * foldSize
                end   = start + trainSize
                
                trainSet = np.arange(start,end) % D
                querySet = np.arange(end, end + querySize) % D
                
                W_train = W[trainSet,:]
                W_query = W[querySet,:]
                
                # Train the model
                model = ctm.newModelAtRandom(W_train, K, dtype=DTYPE)
                queryState = ctm.newQueryState(W_train, model)
                
                plan  = ctm.newTrainPlan(iterations=40, logFrequency=1, fastButInaccurate=useDiagonalPriorCov, debug=True)
                model, queryState, (bndItrs, bndVals, likelies) = ctm.train (W_train, None, model, queryState, plan)
                    
                # Plot the evolution of the bound during training.
                fig, ax1 = plt.subplots()
                ax1.plot(bndItrs, bndVals, 'b-')
                ax1.set_xlabel('Iterations')
                ax1.set_ylabel('Bound', color='b')
                
                ax2 = ax1.twinx()
                ax2.plot(bndItrs, likelies, 'r-')
                ax2.set_ylabel('Likelihood', color='r')
                
                fig.show()
            
                # Plot the topic covariance
                self._plotCov(model)
                
                # Plot the vocab
                plt.imshow(model.vocab, interpolation="none", cmap = cm.Greys_r)
                plt.show()
                
                # Calculating the training set likelihood
                trainLikely.append(ctm.log_likelihood(W_train, model, queryState))
                trainWordCount.append(W_train.data.sum())
                
                # Now query the model.
                plan       = ctm.newTrainPlan(iterations=10, fastButInaccurate=useDiagonalPriorCov)
                queryState = ctm.newQueryState(W_query, model)
                model, queryState = ctm.query(W_query, None, model, queryState, plan)
                
                queryLikely.append(ctm.log_likelihood(W_query, model, queryState))
                queryWordCount.append(W_query.data.sum())
             
            # Print out the likelihood and perplexity for each fold.   
            print ("\n\n\nWith " + ("diagonal" if useDiagonalPriorCov else "full") + " covariances")
            for fold in range(folds):
                trainPerp = np.exp(-trainLikely[fold]/trainWordCount[fold])
                queryPerp = np.exp(-queryLikely[fold]/queryWordCount[fold])
                
                print("Fold %3d: Train-set Likelihood: %12f \t Query-set Likelihood: %12f" % (fold, trainLikely[fold], queryLikely[fold]))
                print("                    Perplexity: %12.2f \t           Perplexity: %12.2f" % (trainPerp, queryPerp))
        
                self.assertTrue(queryPerp < 60.0) # Maximum perplexity.
                self.assertTrue(trainPerp < 60.0)
            print ("\n\n")
            
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
        
        model, query, (bndItrs, bndVals) = ctm.train (W, None, model, queryState, trainPlan)
            
        # Plot the bound
        plt.plot(bndItrs[5:], bndVals[5:])
        plt.xlabel("Iterations")
        plt.ylabel("Variational Bound")
        plt.show()
        
        # Plot the inferred vocab
        plt.imshow(model.vocab, interpolation="none", cmap = cm.Greys_r)
        plt.show()
        
        recons = rowwise_softmax(queryState.means).dot(model.vocab)
        reconsErr = 1./D * np.sum((np.asarray(W.todense()) - recons) * (np.asarray(W.todense()) - recons))
        print ("Final reconstruction error is %f\n\n" % reconsErr)
    

    def testOnRealData(self):
        print ("CTM/Bouchard")
        rd.seed(0xBADB055)
        path = "/Users/bryanfeeney/Desktop/NIPS"
        with open(path + "/ar.pkl", 'rb') as f:
            _, W, _, d = pkl.load(f)
        
        if len(d) == 1:
            d = d[0]
        
        if W.dtype != DTYPE:
            W = W.astype(DTYPE)
            
        docLens   = np.squeeze(np.asarray(W.sum(axis=1)))
        good_rows = (np.where(docLens > 0.5))[0]
        if len(good_rows) < W.shape[0]:
            print ("Some rows in the doc-term matrix are empty. These have been removed.")
        W = W[good_rows, :]
        
        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(W.sum(axis=0)))
        scale = np.reciprocal(1 + freq)
       
        # Initialise the model  
        K = 20
        model      = ctm.newModelAtRandom(W, K, dtype=DTYPE)
        queryState = ctm.newQueryState(W, model)
        trainPlan  = ctm.newTrainPlan(iterations=100, logFrequency=10, fastButInaccurate=False, debug=True)
        
        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = ctm.train (W, None, model, queryState, trainPlan)
        with open(newModelFileFromModel(model), "wb") as f:
            pkl.dump ((model, query, (bndItrs, bndVals, bndLikes)), f)
        
        # Plot the evolution of the bound during training.
        fig, ax1 = plt.subplots()
        ax1.plot(bndItrs, bndVals, 'b-')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Bound', color='b')
        
        ax2 = ax1.twinx()
        ax2.plot(bndItrs, bndLikes, 'r-')
        ax2.set_ylabel('Likelihood', color='r')
        
        fig.show()
        fig.suptitle("CTM/Bouchard (Identity Cov) on NIPS")
        plt.show()
        
        plt.imshow(model.vocab, interpolation="none", cmap = cm.Greys_r)
        plt.show()
    
        # Print out the most likely topic words
        topWordCount = 100
        kTopWordInds = [self.topWordInds(d, model.vocab[k,:] * scale, topWordCount) \
                        for k in range(K)]
        
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