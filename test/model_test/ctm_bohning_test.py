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

from math import ceil

from run.main import modelFile


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
        
    def testOnModelDerivedExample(self):
        print("Cross-validated likelihoods on model-derived example")
        
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
                
                plan  = ctm.newTrainPlan(iterations=50, logFrequency=1, fastButInaccurate=useDiagonalPriorCov)
                model, queryState, (bndItrs, bndVals) = ctm.train (W_train, None, model, queryState, plan)
                    
                # Plot the evoluation of the bound during training.
                plt.plot(bndItrs[5:], bndVals[5:])
                plt.xlabel("Iterations")
                plt.ylabel("Variational Bound")
                plt.show()
            
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
                print("                    Perplexity: %12.2f \t           Perplexity: %12.2f" % (fold, trainPerp, queryPerp))
        
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
    

    def _testOnRealData(self):
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
        trainPlan  = ctm.newTrainPlan(iterations=1000, plot=True, logFrequency=1)
        
        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals) = ctm.train (W, None, model, queryState, trainPlan)
        with open(modelFile(model), "wb") as f:
            pkl.dump ((model, query, (bndItrs, bndVals)), f)
            
        # Plot the bound
        plt.plot(bndItrs[5:], bndVals[5:])
        plt.xlabel("Iterations")
        plt.ylabel("Variational Bound")
        plt.show()
        
        # Print out the most likely topic words, using TF-IDF
        freq, df = self.idf(W)
        idf = D / (1 + df)
        lidf = np.log(idf)
        
        topWordCount = 100
        kTopWordInds = []
        for k in range(K):
            topWordInds = self.topWordInds(d, model.vocab[k,:] * lidf, topWordCount)
            kTopWordInds.append(topWordInds)
        
        print ("Perplexity: %f\n\n" % ctm.perplexity(W, model, query))
        print ("\t\t".join (["Topic " + str(k) for k in range(K)]))
        print ("\n".join ("\t".join (d[kTopWordInds[k][c]] + "\t%0.4f" % model.vocab[k][kTopWordInds[k][c]] for k in range(K)) for c in range(topWordCount)))
        
    def topWords (self, wordDict, vocab, count=10):
        return [wordDict[w] for w in self.topWordInds(wordDict, vocab, count)]

    
    def topWordInds (self, wordDict, vocab, count=10):
        return vocab.argsort()[-count:][::-1]
    
    def idf(self, W):
        '''
        Returns the total corpus word frequency table, and the total
        corpus df counts for each word (i.e. how many documents did
        it occur in
        '''
        counts = W.sum(axis = 0)
        freq = counts.astype(np.float64) / counts.sum()
        freq = np.squeeze(np.asarray(freq))
        freq += 1E-300
        
        w_dat_copy = W.data.copy()
        w_dat_copy[w_dat_copy > 1] = 1
        W2 = ssp.csr_matrix((w_dat_copy, W.indices, W.indptr))

        df = np.squeeze(np.asarray(W2.sum(axis=0)))
        
        return freq, df
    
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